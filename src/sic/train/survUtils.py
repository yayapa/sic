import math
import torch
from pycox.models.utils import pad_col
import pandas as pd
from sic._vendor.torchsurv.loss.cox import baseline_survival_function, survival_function
import numpy as np
import numba

def compute_baseline_rate(self, E: torch.Tensor, T: torch.Tensor) -> float:
    """Compute baseline event rate for a dataset.

    Parameters
    ----------
    E : Tensor (Tq,)
        Event indicators for the dataset.

    T : Tensor (Tq,)
        Time-to-event or censoring times for the dataset.

    Returns
    -------
    float
        Baseline event rate (events per unit time).
    """
    num_events = E.sum().item()
    num_censored = (E == 0).sum().item()
    mean_time = T.mean().item()
    frac_events = num_events / (num_events + num_censored)
    baseline_rate = frac_events / mean_time if mean_time > 0 else 0.0
    return baseline_rate


def pe_targets_ET(E: torch.Tensor, T: torch.Tensor, time_bins: torch.Tensor):
    """
    E,T: (Tq,) for ONE dataset; time_bins: (K+1,) for that dataset
    returns log_dt, is_event: (Tq, K)
    """
    # prepare left/right bin edges for the intervals [left_i, right_i)
    # example: time_bins = [0, 2, 5, 8] -> bins: [0,2), [2,5), [5, + inf)
    # T = [1.5, 7.2, 10.0]
    device = T.device
    left, right = time_bins[:-1], time_bins[1:]
    K = left.numel()
    
    # j = bin index for each 
    # example: j = [0, 2, 2]
    T_exp = T.unsqueeze(-1)                       # (Tq,1)
    in_bin = (T_exp >= left) & (T_exp < right)    # (Tq,K)
    j = in_bin.float().argmax(dim=-1)             # (Tq,)
    overflow = T >= right[-1]
    j[overflow] = K - 1
    
    
    # first build full times 
    full_dt = (right - left).unsqueeze(0).expand(T.shape[0], K)  # (Tq,K)
    arangeK = torch.arange(K, device=device).unsqueeze(0)
    dt = torch.where(arangeK < j.unsqueeze(-1), full_dt, torch.zeros_like(full_dt))

    # then add the partial time in the j-th bin
    t_left_j = left[j]
    dt_j = (T - t_left_j).clamp(min=0.0)
    dt.scatter_(1, j.unsqueeze(-1), dt_j.unsqueeze(-1))
    
    # finally mask with -inf
    dt_output = torch.full_like(dt, float("1e-8")) # avoid log(0)
    pos = dt > 0
    dt_output[pos] = dt[pos] #torch.log(dt[pos])
    # example output dt: [[1.5, 0.0, 0.0], [3.0, 2.0, 2.2], [3.0, 2.0, 5.0]]

    is_event = torch.zeros_like(dt)
    is_event.scatter_(1, j.unsqueeze(-1), E.float().unsqueeze(-1))
    # example E = [1, 0, 1] -> is_event = [[1,0,0], [0,0,0], [0,0,1]]
    return dt_output, is_event

def pe_survival_at_times(log_hz: torch.Tensor, edges: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
    """
    log_hz : (N, K)  log hazards per row
    edges  : (K+1,)  bin edges; last bin is treated as [edges[-2], +inf)
    times  : (M,)    evaluation times; if None -> use right edges edges[1:]
    return: (N, M)   survival S(t) at requested times (or (N, K) if times is None)
    """
    left  = edges[:-1]           
    right = edges[1:]            
    widths = right - left        
    lam = torch.exp(log_hz)     

    # cumulative hazard at finite right edges (up to each bin's right boundary)
    H_edges = torch.cumsum(lam * widths, dim=-1)   # (N, K)

    if times is None:
        # survival at right edges only (finite grid)
        return torch.exp(-H_edges)                 # (N, K)

    # determine bin index for each time (values in [0..K-1]), overflow -> last bin
    idx = torch.searchsorted(right, times).clamp(max=left.numel()-1)  # (M,)

    # cumulative hazard up to the previous bin (sum over m < k)
    idx_m1 = (idx - 1).clamp(min=0)               # (M,)
    H_before = H_edges.gather(1, idx_m1.expand(log_hz.size(0), -1))
    H_before = H_before * (idx > 0).float().unsqueeze(0)

    # within-bin increment using constant hazard in bin k(t)
    left_k = left[idx]                             # (M,)
    lam_k  = lam.gather(1, idx.expand(log_hz.size(0), -1))   # (N, M)
    inc = lam_k * (times - left_k).unsqueeze(0)    # (N, M)

    H_t = H_before + inc
    return torch.exp(-H_t)                         # (N, M)

def build_eval_times(edges: torch.Tensor, T_test: torch.Tensor) -> torch.Tensor:
    """
    edges : (K+1,) sorted bin edges
    T_test: (Nq,) observed test times (unsorted)
    returns strictly increasing, finite, non-negative times <= max(T_test)
    """
    device = edges.device
    dtype  = edges.dtype

    # concat edges' right boundaries with observed times
    times = torch.cat([edges[1:], T_test.to(device, dtype=dtype)], dim=0)

    # keep finite & >= 0, and cap at max observed test time
    tmax  = torch.max(T_test).to(device, dtype=dtype)
    finite = torch.isfinite(times)
    times  = times[finite].clamp(min=0, max=tmax)

    # sort + dedupe (strictly increasing)
    times, _ = torch.sort(times)
    times    = torch.unique(times, sorted=True)

    # ensure at least two distinct points (IBS needs an interval)
    if times.numel() < 2:
        eps = torch.finfo(dtype).eps * (tmax + 1)
        times = torch.tensor([0.0, float(tmax + eps)], device=device, dtype=dtype)

    return times

def compute_bins_per_table(E_train: torch.Tensor, T_train: torch.Tensor, K: int) -> torch.Tensor:
    """
    E_train, T_train: (T_train,) for ONE dataset/table
    Returns time_bins: (K+1,) with time_bins[0]=0 and time_bins[-1] is T_max .
    """
    device = T_train.device
    # only event times 
    ev_times = T_train[E_train.bool()]
    if ev_times.numel() >= 2:  # need at least 2 events to compute percentiles
        pct = torch.linspace(0, 100, K + 1, device=device)
        # torch.quantile works on [0,1], so convert:
        q = pct / 100.0
        edges = torch.quantile(ev_times, q.clamp(0,1))
        edges[0] = 0.0
        edges[-1] = ev_times.max()
    else:
        # raise an error
        raise ValueError("Not enough events in training data to compute time bins.")
        
    # make edges strictly non-decreasing (robustness)
    edges = torch.maximum(edges, torch.cat([edges[:1], edges[:-1] + 1e-12]))
    return edges

def predict_surv_df_deephit(preds, duration_index):
    """Predict the survival function for `input`, i.e., survive all of the event types,
    and return as a pandas DataFrame.
    See `prediction_surv_df` to return a DataFrame instead.

    Arguments:
        input {tuple, np.ndarra, or torch.tensor} -- Input to net.
    
    Returns:
        pd.DataFrame -- Predictions
    """
    surv = predict_surv_deephit(preds)
    return pd.DataFrame(surv, duration_index)

def predict_surv_deephit(preds):
    """Predict the survival function for `input`, i.e., survive all of the event types.
    See `prediction_surv_df` to return a DataFrame instead.

    Arguments:
        input {tuple, np.ndarra, or torch.tensor} -- Input to net.
    
    
    Returns:
        [np.ndarray or tensor] -- Predictions
    """
    cif = predict_cif_deephit(preds)
    surv = 1. - cif.sum(0)
    return surv

def predict_cif_deephit(preds):
    """Predict the cumulative incidence function (cif) for `input`.

    Arguments:
        input {tuple, np.ndarray, or torch.tensor} -- Input to net.

    
    Returns:
        [np.ndarray or tensor] -- Predictions
    """
    pmf = predict_pmf_deephit(preds)
    cif = pmf.cumsum(1)
    return cif

def predict_pmf_deephit(preds):
    """Predict the probability mass fuction (PMF) for `input`.

    Arguments:
        input {tuple, np.ndarray, or torch.tensor} -- Input to net.
    
    
    Returns:
        [np.ndarray or tensor] -- Predictions
    """
    preds = preds.unsqueeze(1) # make sure shape is (N, 1, K)
    pmf = pad_col(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
    pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
    return pmf

def predict_surv_new_times_deephit(surv: torch.Tensor, orig_times: torch.Tensor, new_times: torch.Tensor) -> torch.Tensor:
    """
    surv       : (N, K) survival at orig_times
    orig_times : (K,) original time grid (sorted, finite)
    new_times  : (M,) new time points (sorted, finite)
    returns: (N, M) survival at new_times
    S(t) = S(t_j)*exp(-haz_j*(t - t_j)) where haz_j = 1 - S(t_{j+1})/S(t_j) for t in [t_j, t_{j+1})
    """
    surv_new = torch.zeros((surv.size(0), new_times.size(0)), device=surv.device, dtype=surv.dtype)
    K = orig_times.size(0)
    M = new_times.size(0)
    for j in range(M):
        t_new = new_times[j]
        # find bin index for t_new
        if t_new < orig_times[0]:
            surv_new[:, j] = 1.0
        elif t_new >= orig_times[-1]:
            surv_new[:, j] = surv[:, -1]
        else:
            k = torch.searchsorted(orig_times, t_new).item()
            t_left = orig_times[k-1]
            S_left = surv[:, k-1]
            S_right = surv[:, k]
            haz_k = -torch.log(S_right / S_left) / (orig_times[k] - orig_times[k-1]) # changed
            delta_t = t_new - t_left
            surv_new[:, j] = S_left * torch.exp(-haz_k * delta_t)
    return surv_new

def predict_surv_coxph(log_hazards: torch.Tensor, E_test: torch.Tensor, T_test: torch.Tensor):
    """
    log_hazards : (N, K)  log hazards per row
    E_test      : (N,)    event indicators
    T_test      : (N,)    observed times
    returns: S numpy array (N, K), time_grid numpy array (K,)
    """
    bs = baseline_survival_function(log_hazards, E_test.bool(), T_test)
    H0_vals = -torch.log(bs["baseline_survival"])  # shape (K,)
    time_grid = bs["time"] 
    S = torch.exp(-torch.exp(log_hazards).unsqueeze(1) * H0_vals)
    time_grid = time_grid.cpu().numpy()
    return S, time_grid

def predict_surv_coxph_2(log_hazards_train: torch.Tensor, E_train: torch.Tensor, T_train: torch.Tensor,
                         log_hazards_test: torch.Tensor, T_test: torch.Tensor):
    """
    log_hazards_train : (N_train, K)  log hazards for training data
    E_train          : (N_train,)     event indicators for training data
    T_train          : (N_train,)     observed times for training data
    log_hazards_test : (N_test, K)   log hazards for test data
    E_test           : (N_test,)      event indicators for test data
    T_test           : (N_test,)      observed times for test data
    returns: S numpy array (N_test, K), time_grid numpy array (K,)
    """
    bs = baseline_survival_function(log_hazards_train, E_train.bool(), T_train)
    #print("baseline survival funtion shape:", bs["baseline_survival"].shape)
    #print("log_hazards_test shape:", log_hazards_test.shape)
    #print("log_hazards_train shape:", log_hazards_train.shape)
    S = survival_function(bs, log_hazards_test, T_test)
    #print("survival function shape:", S.shape)
    #print("test times shape:", T_test.shape)

    return S, T_test.cpu().numpy()

def deepeh_loss(out, time, delta):
    """ Taken from the original paper of the EH models"""
    #loss function for Extended hazard model
    eps = 1e-8
    n = len(out[:,0])
    h = 1.30*math.pow(n,-0.2)  ## or 1.59*n^(-1/3)
    time = time.view(n,1)
    delta = delta.view(n,1)
    g1 = out[:,0].view(n,1)
    g2 = out[:,1].view(n,1)
    
    # R = g(Xi) + log(Oi)
    time = torch.clamp(time, min=eps)
    R = torch.add(g1,torch.log(time)) 
    
    S1 =  (delta*g2).sum()/n
    S2 = -(delta*R).sum()/n
    
    # Rj - Ri
    rawones = torch.ones(1,n, device=R.device)
    R1 = torch.mm(R,rawones)
    R2 = torch.mm(torch.t(rawones),torch.t(R))
    DR = R1 - R2 
    
    # K[(Rj-Ri)/h]
    K = normal_density(DR/h)
    Del = torch.mm(delta, rawones)
    DelK = Del*K 
    
    # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
    Dk = torch.sum(DelK, dim=0)/(n*h)  ## Dk would be zero as learning rate too large!

    
    # log {(1/nh) * Deltaj * K[(Rj-Ri)/h]}
    Dk = torch.clamp(Dk, min=eps)  
    log_Dk = torch.log(Dk)    
    
    S3 = (torch.t(delta)*log_Dk).sum()/n    
    
    # Phi((Rj-Ri)/h)
    ncdf=torch.distributions.normal.Normal(torch.tensor([0.0], device=R.device), torch.tensor([1.0], device=R.device)).cdf
    # cauesed nan issue, therefore safe normalization
    DR_normalized = DR/h
    P = ncdf(DR_normalized) 
    L = torch.exp(g2-g1)
    LL = torch.mm(L,rawones)
    LP_sum = torch.sum(LL*P, dim=0)/n
    LP_sum = torch.clamp(LP_sum, min=eps)
    Q = torch.log(LP_sum)
    
    S4 = -(delta*Q.view(n,1)).sum()/n
    
    S = S1 + S2 + S3 + S4  
    S = -S
    return S

def normal_density(a):  
    b = 0.3989423*torch.exp(-0.5*torch.pow(a,2.0))
    return b

def predict_surv_deepeh(out, time, delta, out_new, time_new, delta_new, Riemann_sum_gap=0.1, integral_id=0):
    d = Riemann_sum_gap
    time_new, ind_sort = time_new.sort()
    delta_new = delta_new[ind_sort]
    out_new = out_new[ind_sort,:]
    T_min = torch.min(time).item()
    #T_min = min(min(time), min(time_new)).numpy()
    n_new = len(out_new[:,0])
    
    time_new = time_new.view(n_new,1)
    delta_new = delta_new.view(n_new,1)
    _, ib = out.size()
    expg1_new = torch.exp(out_new[:,0].view(n_new,1)).to(device=time.device)
    
    if ib == 2:
        expg2_new = torch.exp(out_new[:,1].view(n_new,1)).to(device=time.device)
    
    ####################################
    ind_si = torch.zeros(n_new,dtype=int)
    T0 = T_min   
    a = (time_new[0].item()-T_min)# (time_new[0].numpy()-T_min)/2

    if d < a:
         TT0 = np.arange(T_min, time_new[0].item(), min(a,d))
         n0 = len(TT0)-1
         dt0 = min(a,d)*np.ones(len(TT0))
    else:
         TT0 = np.append(T_min, time_new[0].item())
         n0 = 1
         dt0 = np.append(T_min, a)
        
     
    for k in range(n_new):
         T1 = time_new[k].item()
         dd = (T1 - T0)
    
         if k == 0:
             ind_si[k] = n0
             T = TT0
             dt = dt0
         else:
             if dd == 0:
                 ind_si[k] = ind_si[k-1]
             else:
                 if (dd <= d):
                     ind_si[k] = ind_si[k-1] + 1
                     T = np.append(T,T1)
                     dt = np.append(dt,dd)
                 else:
                     TT0 = np.arange(T0, T1, d)
                     dt0 = d*np.ones(len(TT0))
                     n0 = len(TT0)-1
                     ind_si[k] = ind_si[k-1] + n0
                     T = np.append(T,TT0[1:])
                     dt = np.append(dt,dt0[1:])                   
         T0 = T1 
     
    m = len(T)
    T = torch.from_numpy(T).view(1,m)
    T = T.type_as(time)
    T = T.to(device=time.device)
    dt = torch.from_numpy(dt).view(1,m) 
    dt = dt.type_as(time)
    dt = dt.to(device=time.device)
     
    if (integral_id == 0):#(integral_id == 'gbsg')|(integral_id == 'metabric'):
         mat_si = torch.zeros(n_new,n_new) 
         #T_med = torch.zeros(n_new,1)
         
         #for i in range(n_new):
         for i in numba.prange(n_new):
            A = (torch.exp(-expg2_new[i,0]*torch.cumsum(dt*hazard_fun(out, time, delta, expg1_new[i,0]*T), dim=1)))
            mat_si[i,:] = A[0,ind_si]
            #T_med[i,:]= max(T[A>=0.5])
                            
         #ss= mat_si.detach().numpy()    
         ss = mat_si
         tt = time_new.view(-1).cpu().numpy()     # shape (n_new,)
         dlt = delta_new.view(-1).cpu().numpy()   # shape (n_new,) 
                                     
    return ss, tt, dlt, ind_sort


def hazard_fun(out, time, delta, t):
    # t is a vector of input evaluated time
    n = len(out[:,0])
    h = 1.30*math.pow(n,-0.2)  ## or 1.59*n^(-1/3)
    time = time.view(n,1)
    delta = delta.view(n,1)
    ia, ib = out.size()
    g1 = out[:,0].view(n,1)
    if ib == 2:
        g2 = out[:,1].view(n,1)  
    else:
        g2 = g1
    
    # R = g(Xi) + log(Oi)
    R = torch.add(g1,torch.log(time)) 
    
    # Rj - Ri
    p = max(len(t),len(torch.t(t)))
    T = t 
    T = T.view(1,p)
    log_t = torch.log(T)

    # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
    Dk = torch.sum(torch.mm(delta, torch.ones([1,p], dtype=out.dtype, device=out.device))\
                   *normal_density((torch.mm(R,torch.ones([1,p], dtype=out.dtype, device=out.device)) - torch.mm(torch.ones([n,1], dtype=out.dtype, device=out.device), log_t))/h)/h, dim=0)/n
    
    # Phi((Rj-Ri)/h)
    ncdf=torch.distributions.normal.Normal(torch.tensor([0.0], dtype=out.dtype, device=out.device), torch.tensor([1.0], dtype=out.dtype, device=out.device)).cdf
    L = torch.exp(g2-g1)

    dominator = (torch.sum((torch.mm(L,torch.ones([1,p], dtype=out.dtype, device=out.device)))*\
                           ncdf((torch.mm(R,torch.ones([1,p], dtype=out.dtype, device=out.device)) - torch.mm(torch.ones([n,1], dtype=out.dtype, device=out.device), log_t))/h), dim=0)/n)*T  #LP_sum*T
    lam = Dk/dominator  
    return lam

### helper function  
def normal_density(a):  
    b = 0.3989*torch.exp(-0.5*torch.pow(a,2.0))
    return b  

def is_comparable_antolini(t_i, t_j, d_i, d_j):
    a = (t_i < t_j) * d_i
    b = (t_i == t_j)*d_i*(d_j == 0.)
    c = max(a,b)
    return c

def is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):  
    return (s_i < s_j) * is_comparable_antolini(t_i, t_j, d_i, d_j)

def is_concordant_dominant_antolini(t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i <= t_j:
        conc = (t_i <= t_j)* 1.
    return conc * is_comparable_antolini(t_i, t_j, d_i, d_j)

def sum_concordant_antolini(s, t, d):
    n = len(t)
    count = 0.
    count_d = 0.
    for i in range(n):
        for j in range(n):
            if j != i:
                count += is_concordant_antolini(s[i, i], s[j, i], t[i], t[j], d[i], d[j])
                count_d += is_concordant_dominant_antolini(t[i], t[j], d[i], d[j])
    ratio = count/count_d
    return ratio

def cox_c_index(out_new, time_new, delta_new):
    n_new = len(out_new[:,0])
    rawones = torch.ones(1,n_new)
    s = torch.mm(out_new, rawones)
    s = -s.detach().numpy()
    t = time_new.detach().numpy()
    d = delta_new.detach().numpy()
    
    return sum_concordant_antolini(s, t, d)   
    

def sort_back_surv(ind_sort, ss):
    ss_sorted_back = torch.zeros_like(ss, device=ss.device, dtype=ss.dtype)
    for i in range(len(ind_sort)):
        ss_sorted_back[ind_sort[i],:] = ss[i,:]
    return ss_sorted_back


def d_calibration(points, is_alive, nbins=20, differentiable=False, gamma=1.0, device='cpu'):
    # each "point" in points is a time for datapoint i mapped through the model CDF
    # each such time_i is a survival time if not censored or a time sampled
    # uniformly in (censor time, max time
    # compute empirical cdf of cdf-mapped-times
    # Move censored points with cdf values greater than 1 - 1e-4 t0 uncensored group
    new_is_alive = is_alive.detach().clone()
    new_is_alive[points > 1. - 1e-4] = 0

    points = points.to(device).view(-1, 1).float()
    # print(points[:200])
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    bin_width = 1.0/nbins
    bin_indices = torch.arange(nbins).view(1,-1).float().to(device)
    bin_a = bin_indices * bin_width #+ 0.02*torch.rand(size=bin_indices.shape)
    noise = 1e-6/nbins*torch.rand(size=bin_indices.shape).to(device)
    if not differentiable:
        noise = noise * 0.
    cum_noise = torch.cumsum(noise, dim=1)
    bin_width = torch.tensor([bin_width]*nbins).to(device) + cum_noise
    bin_b = bin_a + bin_width

    bin_b_max = bin_b[:,-1]
    bin_b = bin_b/bin_b_max
    bin_a[:,1:] = bin_b[:,:-1]
    bin_width = bin_b - bin_a

    # CENSORED POINTS
    points_cens = points[new_is_alive.long()==1]
    upper_diff_for_soft_cens = bin_b - points_cens
    # To solve optimization issue, we change the first left bin boundary to be -1.;
    # we change the last right bin boundary to be 2.
    bin_b[:,-1] = 2.
    bin_a[:,0] = -1.
    lower_diff_cens = points_cens - bin_a # p - a
    upper_diff_cens = bin_b - points_cens # b - p
    diff_product_cens = lower_diff_cens * upper_diff_cens
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        bin_index_ohe = torch.sigmoid(gamma * diff_product_cens)
        exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)
    else:
        # (p-a)*(b-p)
        bin_index_ohe = (lower_diff_cens >= 0).float() * (upper_diff_cens > 0).float()
        exact_bins_next = (lower_diff_cens <= 0).float()  # all bins after correct bin

    EPS = 1e-13
    right_censored_interval_size = 1 - points_cens + EPS

    # each point's distance from its bin's upper limit
    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)

    # assigns weights to each full bin that is larger than the point
    # full_bin_assigned_weight = exact_bins*bin_width
    # 1 / right_censored_interval_size is the density of the uniform over [F(c),1]
    full_bin_assigned_weight = (exact_bins_next*bin_width.view(1,-1)/right_censored_interval_size.view(-1,1)).sum(0)
    partial_bin_assigned_weight = (upper_diff_within_bin/right_censored_interval_size).sum(0)
    assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    points_uncens = points[new_is_alive.long() == 0]
    # compute p - a and b - p
    lower_diff = points_uncens - bin_a
    upper_diff = bin_b - points_uncens
    diff_product = lower_diff * upper_diff
    assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
    assert lower_diff.shape == (points_uncens.shape[0], bin_a.shape[1])
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        soft_membership = torch.sigmoid(gamma*diff_product)
        fraction_in_bins = soft_membership.sum(0)
        # print('soft_membership', soft_membership)
    else:
        # (p-a)*(b-p)
        exact_membership = (lower_diff >= 0).float() * (upper_diff > 0).float()
        fraction_in_bins = exact_membership.sum(0)

    assert fraction_in_bins.shape == (nbins, ), fraction_in_bins.shape

    frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) /points.shape[0]
    return torch.pow(frac_in_bins - bin_width, 2).sum()