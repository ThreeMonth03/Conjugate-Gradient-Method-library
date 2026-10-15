Execution reuse and launch evidence — 2026-10-16
================================================

This report retains every measured native result, every strict framework control, the one fixed CPU confirmation and all remaining CUDA losses. Fresh API, explicit Session reuse and prepared plans are separate allocation contracts. No universal improvement or complete strict-gate pass follows from selected medians.

The original CPU ABBA retains **3 native strict regressions**; its whole-batch verdict is **regression**. The single fixed CPU confirmation's pooled verdict is **inconclusive**, without replacing the original report. CUDA launch native rows: 21 within 5%, 3 inconclusive, 0 regression, 0 comparison failed. In the separately scoped device Session characterization, native point medians are lower than same-artifact Torch in **24/24 rows**; every remaining loss and every host/API/prepared row is retained below. Those different-scope point estimates are descriptive, not strict speedup evidence.

.. csv-table:: Actual measured hardware/compiler identities; full package/driver metadata stays in raw JSON
   :class: benchmark-results
   :header: "cohort","CPU","platform","native host compiler","host flags","NumPy","Torch","CUDA device","CUDA compiler","CUDA architectures"

   "CPU ABBA","13th Gen Intel(R) Core(TM) i7-13700K","Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39","GNU 16.0.1"," -O3 -DNDEBUG","2.2.6","2.8.0+cpu","none","none","none"
   "CUDA final scopes","13th Gen Intel(R) Core(TM) i7-13700K","Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39","GNU 13.3.0"," -O3 -DNDEBUG","2.2.6","2.8.0+cu128","NVIDIA GeForce RTX 2060","NVIDIA 12.8.93","75"

Measurement and decision rules
------------------------------

The measured host is an i7-13700K under WSL2 with an RTX 2060. CPU and
CUDA use separate project-local Python environments. Affinity 0 is a logical
CPU constraint, not physical-core isolation. Builds, tests and other project
numerical work run outside the quiet campaigns. Windows/display activity,
GPU clocks, temperature and other host background work are not controlled.
Actual compiler, flags, runtime, device identity and library pools remain in
every raw artifact; CPU and CUDA compiler identities are separate.

Inputs are FP64, seed 20261015, rtol=1e-8, atol=1e-12, maxiter=2000,
standard CG/PCG with residual_refresh=50. The native mode and dense provider
are always explicit. Each process uses three warmups, 30 complete balanced
Williams rounds, three repetitions per dense round or one per CSR round.
Every invocation charges the initial-vector reset. A CUDA timing block fences
before the block and after the last invocation, including host dispatch,
iteration/control synchronization and completed GPU work. Input transfers
and independent CPU verification downloads are outside timing. These are
resident-input solves, not end-to-end host-to-host comparisons.

Strict A/B comparisons preserve full scope/options, fixture hashes and each
candidate's iterations/matvecs. No timing normalization or filtered projection
bypasses a work mismatch. Reported A/B medians pool rounds across artifact
blocks; brackets show the range of process medians. The saved hierarchical
bootstrap resamples artifact blocks, then rounds, 10,000 times with seed
20261015 and 95% percentile intervals. The fixed material-regression budget
is 5%: upper ratio bound <=1.05 supports no material regression; lower bound
>1.05 signals regression; overlap is inconclusive. This budget does not prove
a speedup. Per-case intervals have no simultaneous-coverage correction;
serial dependence and few process blocks limit precision. Reference controls
can also drift and are retained without subtracting their noise from native
results. Numerical range failures are never reclassified as accepted solves.

The three-scope characterization instead reports median-of-process-medians
and process min–max ranges. Different estimators are explicitly kept apart.
No strict confidence interval is assigned to a different scope, host/device
engine or framework implementation. Rounding-induced work differences remain
visible as time to independently accepted tolerance. PyTorch runs this
project's eager CG/PCG recurrence, not an official Torch CG solver; Torch's
caching allocator and provider kernels remain part of that candidate.

CPU GEMV/reduction implementation; original ABBA
------------------------------------------------

The CPU change targets the unfused (``fused=False``) GEMV path; fused kernels
remain unchanged. All 16 unfused native rows support no material regression
with identical iteration/matvec counts. At n=2048, current/baseline medians
are 0.579–0.676, or 32–42% lower, across none/Jacobi and prepared/API.

The three original native regressions occur in fused none/API at n=64/256/1024;
unchanged NumPy/Torch controls regress in those same cells. All observations
remain reported, without causal attribution or subtracting control drift.
The original whole-batch verdict remains regression and the fixed pooled
confirmation remains inconclusive.

Whole-batch strict verdict: **regression**. Native rows: 18 within 5%, 11 inconclusive, 3 regression, 0 comparison failed. Every original result remains in this table; a lower point estimate alone is not a passed strict gate.

``numpy-out`` controls: 6 within 5%, 23 inconclusive, 3 regression, 0 comparison failed.

``torch-eager-out`` controls: 6 within 5%, 23 inconclusive, 3 regression, 0 comparison failed.

.. csv-table:: Every native A/B result
   :class: benchmark-results
   :header: "configuration","case","candidate","A ms [process range]","B ms [process range]","B/A","95% ratio interval","strict verdict","A iterations/matvecs","B iterations/matvecs","B true residual","threshold"

   "none/prepared/fused=true","dense-64","native","0.0291 [0.0282–0.0301]","0.0301 [0.0298–0.0304]","1.035","[0.956, 1.119]","inconclusive","49/51","49/51","2.072e-06","2.738e-06"
   "none/prepared/fused=true","dense-256","native","0.4288 [0.4246–0.4484]","0.4246 [0.4130–0.4281]","0.990","[0.902, 1.031]","within 5%","67/70","67/70","5.707e-06","6.342e-06"
   "none/prepared/fused=true","dense-1024","native","9.7119 [9.4245–9.8641]","9.5176 [9.4731–9.6779]","0.980","[0.949, 1.037]","within 5%","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=true","dense-2048","native","127.1318 [119.2156–134.5177]","133.9246 [126.9155–142.4160]","1.053","[0.929, 1.204]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/api/fused=true","dense-64","native","0.0401 [0.0395–0.0405]","0.0597 [0.0590–0.0622]","1.488","[1.268, 1.738]","regression","49/51","49/51","2.072e-06","2.738e-06"
   "none/api/fused=true","dense-256","native","0.4455 [0.4368–0.4600]","0.5573 [0.5491–0.6153]","1.251","[1.159, 1.444]","regression","67/70","67/70","5.707e-06","6.342e-06"
   "none/api/fused=true","dense-1024","native","8.5877 [8.4165–8.7181]","11.6586 [11.5896–11.6967]","1.358","[1.310, 1.420]","regression","70/73","70/73","9.505e-06","1.079e-05"
   "none/api/fused=true","dense-2048","native","141.5111 [128.6857–192.7386]","148.5604 [146.0603–151.5539]","1.050","[0.740, 1.195]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "jacobi/prepared/fused=true","dense-64","native","0.0069 [0.0068–0.0069]","0.0078 [0.0072–0.0100]","1.135","[1.012, 1.734]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=true","dense-256","native","0.0402 [0.0396–0.0405]","0.0448 [0.0394–0.0491]","1.115","[0.968, 1.250]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=true","dense-1024","native","1.1747 [1.0770–1.2249]","1.2021 [1.0520–1.4456]","1.023","[0.834, 1.381]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=true","dense-2048","native","10.1645 [9.3780–10.5255]","10.6621 [9.9647–11.6004]","1.049","[0.931, 1.253]","inconclusive","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/api/fused=true","dense-64","native","0.0136 [0.0136–0.0136]","0.0133 [0.0133–0.0137]","0.981","[0.893, 1.070]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/api/fused=true","dense-256","native","0.0470 [0.0454–0.0479]","0.0503 [0.0487–0.0522]","1.070","[0.977, 1.194]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/api/fused=true","dense-1024","native","0.7500 [0.7497–0.7564]","0.7809 [0.7159–0.8304]","1.041","[0.930, 1.135]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/api/fused=true","dense-2048","native","8.9574 [8.1346–9.9082]","8.4909 [8.1763–8.8082]","0.948","[0.793, 1.090]","inconclusive","3/5","3/5","1.245e-05","1.474e-05"
   "none/prepared/fused=false","dense-64","native","0.0594 [0.0588–0.0602]","0.0321 [0.0316–0.0334]","0.541","[0.517, 0.588]","within 5%","49/51","49/51","2.072e-06","2.738e-06"
   "none/prepared/fused=false","dense-256","native","1.4503 [1.4477–1.4532]","0.4429 [0.4417–0.4453]","0.305","[0.295, 0.314]","within 5%","67/70","67/70","5.707e-06","6.342e-06"
   "none/prepared/fused=false","dense-1024","native","30.7270 [30.3756–30.8826]","10.2126 [10.0784–10.4172]","0.332","[0.322, 0.346]","within 5%","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=false","dense-2048","native","207.4739 [206.2917–208.1231]","140.2050 [138.7118–141.5010]","0.676","[0.647, 0.712]","within 5%","70/73","70/73","1.323e-05","1.474e-05"
   "none/api/fused=false","dense-64","native","0.0676 [0.0671–0.0683]","0.0433 [0.0408–0.0467]","0.641","[0.575, 0.780]","within 5%","49/51","49/51","2.072e-06","2.738e-06"
   "none/api/fused=false","dense-256","native","1.4852 [1.4782–1.4964]","0.4535 [0.4535–0.4557]","0.305","[0.291, 0.332]","within 5%","67/70","67/70","5.707e-06","6.342e-06"
   "none/api/fused=false","dense-1024","native","30.4321 [30.3481–30.4964]","8.6828 [8.6804–8.8068]","0.285","[0.280, 0.298]","within 5%","70/73","70/73","9.505e-06","1.079e-05"
   "none/api/fused=false","dense-2048","native","215.6052 [213.8326–215.6124]","131.6490 [130.8589–140.3086]","0.611","[0.590, 0.684]","within 5%","70/73","70/73","1.323e-05","1.474e-05"
   "jacobi/prepared/fused=false","dense-64","native","0.0111 [0.0106–0.0115]","0.0079 [0.0078–0.0081]","0.716","[0.643, 0.789]","within 5%","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=false","dense-256","native","0.1366 [0.1359–0.1368]","0.0425 [0.0421–0.0431]","0.311","[0.300, 0.328]","within 5%","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=false","dense-1024","native","2.7272 [2.5893–2.9558]","1.1113 [1.0821–1.1378]","0.407","[0.359, 0.444]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=false","dense-2048","native","17.1767 [15.7812–17.8694]","9.9477 [9.7127–9.9914]","0.579","[0.528, 0.642]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/api/fused=false","dense-64","native","0.0186 [0.0178–0.0225]","0.0145 [0.0142–0.0152]","0.776","[0.606, 0.877]","within 5%","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/api/fused=false","dense-256","native","0.1626 [0.1497–0.1731]","0.0548 [0.0502–0.0632]","0.337","[0.287, 0.423]","within 5%","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/api/fused=false","dense-1024","native","2.8870 [2.8213–2.9104]","0.8120 [0.7768–0.8830]","0.281","[0.259, 0.321]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/api/fused=false","dense-2048","native","14.6047 [14.1249–15.0368]","8.7879 [8.4863–8.8939]","0.602","[0.551, 0.638]","within 5%","3/5","3/5","1.245e-05","1.474e-05"

.. csv-table:: All framework controls
   :class: benchmark-results
   :header: "configuration","case","candidate","A ms [process range]","B ms [process range]","B/A","95% ratio interval","strict verdict","A iterations/matvecs","B iterations/matvecs","B true residual","threshold"

   "none/prepared/fused=true","dense-64","numpy-out","0.2070 [0.1978–0.2116]","0.2050 [0.2014–0.2080]","0.990","[0.933, 1.060]","inconclusive","49/51","49/51","2.047e-06","2.738e-06"
   "none/prepared/fused=true","dense-64","torch-eager-out","0.3039 [0.2944–0.3196]","0.3165 [0.3032–0.3272]","1.041","[0.936, 1.166]","inconclusive","49/51","49/51","2.447e-06","2.738e-06"
   "none/prepared/fused=true","dense-256","numpy-out","0.5982 [0.5646–0.6203]","0.6746 [0.6525–0.7000]","1.128","[1.042, 1.250]","inconclusive","67/70","67/70","5.647e-06","6.342e-06"
   "none/prepared/fused=true","dense-256","torch-eager-out","0.7503 [0.7360–0.7915]","0.7385 [0.7210–0.7538]","0.984","[0.870, 1.060]","inconclusive","67/70","67/70","5.726e-06","6.342e-06"
   "none/prepared/fused=true","dense-1024","numpy-out","9.5957 [9.2636–10.1367]","9.5590 [9.5064–9.7983]","0.996","[0.924, 1.066]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=true","dense-1024","torch-eager-out","10.3126 [9.8846–10.6704]","10.1459 [9.8721–10.2397]","0.984","[0.918, 1.073]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=true","dense-2048","numpy-out","99.5133 [88.7630–106.0674]","116.9145 [106.2574–119.9971]","1.175","[0.997, 1.371]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/prepared/fused=true","dense-2048","torch-eager-out","114.3616 [105.2146–120.5673]","123.3307 [123.3307–123.5492]","1.078","[0.965, 1.231]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/api/fused=true","dense-64","numpy-out","0.2156 [0.2097–0.2191]","0.2995 [0.2868–0.3111]","1.389","[1.287, 1.511]","regression","49/51","49/51","2.047e-06","2.738e-06"
   "none/api/fused=true","dense-64","torch-eager-out","0.3177 [0.3177–0.3193]","0.5164 [0.4869–0.5495]","1.625","[1.396, 1.768]","regression","49/51","49/51","2.447e-06","2.738e-06"
   "none/api/fused=true","dense-256","numpy-out","0.5829 [0.5718–0.5961]","0.9308 [0.9243–0.9365]","1.597","[1.452, 1.741]","regression","67/70","67/70","5.647e-06","6.342e-06"
   "none/api/fused=true","dense-256","torch-eager-out","0.7805 [0.7775–0.8009]","1.4283 [1.2537–1.4588]","1.830","[1.473, 1.932]","regression","67/70","67/70","5.726e-06","6.342e-06"
   "none/api/fused=true","dense-1024","numpy-out","8.3408 [8.3106–8.3671]","11.9604 [11.9437–12.0328]","1.434","[1.405, 1.474]","regression","70/73","70/73","9.505e-06","1.079e-05"
   "none/api/fused=true","dense-1024","torch-eager-out","9.0324 [9.0024–9.0440]","13.6667 [13.3234–13.9336]","1.513","[1.447, 1.572]","regression","70/73","70/73","9.505e-06","1.079e-05"
   "none/api/fused=true","dense-2048","numpy-out","123.2393 [109.5910–170.6077]","126.1648 [118.9971–130.9053]","1.024","[0.685, 1.204]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/api/fused=true","dense-2048","torch-eager-out","128.8475 [111.9851–177.8665]","133.2149 [125.2021–135.2453]","1.034","[0.709, 1.226]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "jacobi/prepared/fused=true","dense-64","numpy-out","0.0298 [0.0297–0.0298]","0.0385 [0.0309–0.0419]","1.293","[1.014, 1.466]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=true","dense-64","torch-eager-out","0.0476 [0.0456–0.0486]","0.0660 [0.0507–0.0755]","1.386","[1.003, 1.689]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=true","dense-256","numpy-out","0.0562 [0.0525–0.0581]","0.0746 [0.0610–0.0850]","1.327","[1.041, 1.643]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=true","dense-256","torch-eager-out","0.0720 [0.0681–0.0756]","0.0974 [0.0722–0.1111]","1.354","[0.943, 1.734]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=true","dense-1024","numpy-out","1.1417 [1.0622–1.1551]","1.1669 [1.0752–1.3057]","1.022","[0.916, 1.243]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=true","dense-1024","torch-eager-out","1.3050 [1.2236–1.4066]","1.3659 [1.2101–1.5161]","1.047","[0.843, 1.275]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=true","dense-2048","numpy-out","8.6770 [7.2163–9.4465]","8.5828 [8.5102–8.6211]","0.989","[0.862, 1.263]","inconclusive","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/prepared/fused=true","dense-2048","torch-eager-out","9.5979 [9.4643–9.7681]","9.0384 [8.7548–9.1717]","0.942","[0.871, 1.040]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/api/fused=true","dense-64","numpy-out","0.0405 [0.0403–0.0409]","0.0409 [0.0406–0.0413]","1.009","[0.956, 1.039]","within 5%","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/api/fused=true","dense-64","torch-eager-out","0.0708 [0.0697–0.0721]","0.0713 [0.0702–0.0721]","1.007","[0.944, 1.062]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/api/fused=true","dense-256","numpy-out","0.0666 [0.0638–0.0686]","0.0642 [0.0621–0.0719]","0.963","[0.890, 1.146]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/api/fused=true","dense-256","torch-eager-out","0.1039 [0.1008–0.1064]","0.1033 [0.1003–0.1097]","0.994","[0.904, 1.134]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/api/fused=true","dense-1024","numpy-out","0.7926 [0.7659–0.8210]","0.8254 [0.7618–0.8591]","1.041","[0.923, 1.138]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/api/fused=true","dense-1024","torch-eager-out","0.9153 [0.8671–0.9376]","0.9572 [0.9275–1.0125]","1.046","[0.954, 1.214]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/api/fused=true","dense-2048","numpy-out","8.1583 [7.5331–8.5854]","7.2204 [7.0783–7.4492]","0.885","[0.758, 1.001]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/api/fused=true","dense-2048","torch-eager-out","7.5759 [7.1599–8.3575]","7.7591 [7.6395–7.8702]","1.024","[0.878, 1.129]","inconclusive","3/5","3/5","1.245e-05","1.474e-05"
   "none/prepared/fused=false","dense-64","numpy-out","0.2076 [0.2070–0.2085]","0.2070 [0.2052–0.2090]","0.997","[0.951, 1.027]","within 5%","49/51","49/51","2.047e-06","2.738e-06"
   "none/prepared/fused=false","dense-64","torch-eager-out","0.3270 [0.3266–0.3305]","0.3042 [0.3006–0.3078]","0.930","[0.893, 0.984]","within 5%","49/51","49/51","2.447e-06","2.738e-06"
   "none/prepared/fused=false","dense-256","numpy-out","0.6467 [0.6430–0.6527]","0.6460 [0.5986–0.6982]","0.999","[0.906, 1.109]","inconclusive","67/70","67/70","5.647e-06","6.342e-06"
   "none/prepared/fused=false","dense-256","torch-eager-out","0.7662 [0.7662–0.7665]","0.7931 [0.7752–0.8025]","1.035","[0.971, 1.105]","inconclusive","67/70","67/70","5.726e-06","6.342e-06"
   "none/prepared/fused=false","dense-1024","numpy-out","9.6859 [9.5465–9.7462]","9.9686 [9.5383–10.2351]","1.029","[0.959, 1.094]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=false","dense-1024","torch-eager-out","10.4888 [10.4065–10.7140]","10.6948 [10.5103–11.0233]","1.020","[0.975, 1.081]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=false","dense-2048","numpy-out","95.9973 [91.8661–99.1882]","109.5718 [94.8426–119.9772]","1.141","[0.931, 1.351]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/prepared/fused=false","dense-2048","torch-eager-out","111.9841 [110.1135–113.2550]","120.7513 [118.7305–123.6690]","1.078","[1.014, 1.151]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/api/fused=false","dense-64","numpy-out","0.2134 [0.2107–0.2146]","0.2050 [0.2010–0.2120]","0.961","[0.922, 1.029]","within 5%","49/51","49/51","2.047e-06","2.738e-06"
   "none/api/fused=false","dense-64","torch-eager-out","0.3175 [0.3131–0.3205]","0.3247 [0.3232–0.3262]","1.023","[0.937, 1.077]","inconclusive","49/51","49/51","2.447e-06","2.738e-06"
   "none/api/fused=false","dense-256","numpy-out","0.6246 [0.5751–0.6536]","0.6426 [0.5955–0.6815]","1.029","[0.895, 1.196]","inconclusive","67/70","67/70","5.647e-06","6.342e-06"
   "none/api/fused=false","dense-256","torch-eager-out","0.8112 [0.7252–0.8365]","0.8149 [0.7859–0.8316]","1.005","[0.935, 1.157]","inconclusive","67/70","67/70","5.726e-06","6.342e-06"
   "none/api/fused=false","dense-1024","numpy-out","8.4386 [8.4287–8.4673]","8.6116 [8.4711–8.8094]","1.021","[0.991, 1.066]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "none/api/fused=false","dense-1024","torch-eager-out","8.9501 [8.9271–9.0104]","9.2984 [9.2866–9.3233]","1.039","[1.008, 1.067]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "none/api/fused=false","dense-2048","numpy-out","118.3176 [115.1932–121.1304]","111.8489 [109.4148–117.0682]","0.945","[0.871, 1.068]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "none/api/fused=false","dense-2048","torch-eager-out","127.0009 [126.6840–128.8439]","116.0280 [112.9043–128.4565]","0.914","[0.861, 1.066]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "jacobi/prepared/fused=false","dense-64","numpy-out","0.0305 [0.0299–0.0328]","0.0311 [0.0305–0.0321]","1.018","[0.898, 1.136]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=false","dense-64","torch-eager-out","0.0498 [0.0482–0.0523]","0.0514 [0.0510–0.0534]","1.031","[0.927, 1.135]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=false","dense-256","numpy-out","0.0560 [0.0520–0.0638]","0.0608 [0.0607–0.0613]","1.085","[0.903, 1.227]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=false","dense-256","torch-eager-out","0.0795 [0.0756–0.0823]","0.0734 [0.0728–0.0736]","0.924","[0.842, 1.027]","within 5%","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=false","dense-1024","numpy-out","1.0858 [1.0473–1.1360]","1.0067 [0.9510–1.1347]","0.927","[0.786, 1.124]","inconclusive","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=false","dense-1024","torch-eager-out","1.3029 [1.2712–1.3200]","1.1817 [1.1512–1.2294]","0.907","[0.839, 1.039]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=false","dense-2048","numpy-out","9.5583 [9.4604–10.0606]","7.5049 [7.3675–7.5702]","0.785","[0.679, 0.849]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/prepared/fused=false","dense-2048","torch-eager-out","9.7048 [9.2465–9.8699]","8.3512 [7.8973–9.1199]","0.861","[0.761, 1.052]","inconclusive","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/api/fused=false","dense-64","numpy-out","0.0450 [0.0402–0.0536]","0.0437 [0.0413–0.0466]","0.970","[0.756, 1.188]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/api/fused=false","dense-64","torch-eager-out","0.0884 [0.0770–0.1063]","0.0748 [0.0704–0.0860]","0.846","[0.658, 1.200]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/api/fused=false","dense-256","numpy-out","0.0824 [0.0754–0.0949]","0.0823 [0.0706–0.1040]","0.998","[0.739, 1.401]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/api/fused=false","dense-256","torch-eager-out","0.1448 [0.1166–0.1767]","0.1342 [0.1068–0.1720]","0.927","[0.600, 1.480]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/api/fused=false","dense-1024","numpy-out","0.9938 [0.9002–1.0899]","0.8287 [0.7866–0.8551]","0.834","[0.707, 0.979]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/api/fused=false","dense-1024","torch-eager-out","1.2149 [1.1030–1.2569]","1.0024 [0.9609–1.0218]","0.825","[0.743, 0.973]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/api/fused=false","dense-2048","numpy-out","7.3948 [6.6474–8.4181]","7.1452 [6.8059–7.8433]","0.966","[0.786, 1.186]","inconclusive","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/api/fused=false","dense-2048","torch-eager-out","8.7637 [8.3173–9.3076]","7.6505 [7.4301–7.8587]","0.873","[0.774, 0.976]","within 5%","3/5","3/5","1.245e-05","1.474e-05"

Changed same-candidate work rows: **0**; no per-iteration normalization.

CPU current native versus matched NumPy/Torch controls
------------------------------------------------------

Across all **32 current native rows** (eight complete configurations, including both fusion flags), **12/32** native medians remain at or above matched NumPy and **8/32** remain at or above matched Torch. Each comparison uses the same configuration, case and current-side artifacts. Medians pool all rounds in the two current artifact blocks, exactly as the A/B tables above; this is not the separate median-of-process-medians estimator. These framework point comparisons have no computed confidence interval or strict speedup verdict. All matched control timings and work counts remain in the preceding tables.

.. csv-table:: Matched current-side framework summary; ratios are native/reference
   :class: benchmark-results
   :header: "fusion","reference","native rows","lower native median","native median at/above reference","native/reference range","different work"

   "fused=true","numpy-out","16","11","5","0.147–1.242","0"
   "fused=true","torch-eager-out","16","12","4","0.095–1.180","0"
   "fused=false","numpy-out","16","9","7","0.155–1.325","0"
   "fused=false","torch-eager-out","16","12","4","0.106–1.191","0"
   "both flags","numpy-out","32","20","12","0.147–1.325","0"
   "both flags","torch-eager-out","32","24","8","0.095–1.191","0"

.. csv-table:: Every remaining CPU framework point loss; no candidate/case filtering
   :class: benchmark-results
   :header: "configuration","case","reference","native ms [process range]","reference ms [process range]","native/reference","higher native median %"

   "none/prepared/fused=true","dense-2048","numpy-out","133.9246 [126.9155–142.4160]","116.9145 [106.2574–119.9971]","1.145","14.55"
   "none/prepared/fused=true","dense-2048","torch-eager-out","133.9246 [126.9155–142.4160]","123.3307 [123.3307–123.5492]","1.086","8.59"
   "none/api/fused=true","dense-2048","numpy-out","148.5604 [146.0603–151.5539]","126.1648 [118.9971–130.9053]","1.178","17.75"
   "none/api/fused=true","dense-2048","torch-eager-out","148.5604 [146.0603–151.5539]","133.2149 [125.2021–135.2453]","1.115","11.52"
   "jacobi/prepared/fused=true","dense-1024","numpy-out","1.2021 [1.0520–1.4456]","1.1669 [1.0752–1.3057]","1.030","3.02"
   "jacobi/prepared/fused=true","dense-2048","numpy-out","10.6621 [9.9647–11.6004]","8.5828 [8.5102–8.6211]","1.242","24.23"
   "jacobi/prepared/fused=true","dense-2048","torch-eager-out","10.6621 [9.9647–11.6004]","9.0384 [8.7548–9.1717]","1.180","17.96"
   "jacobi/api/fused=true","dense-2048","numpy-out","8.4909 [8.1763–8.8082]","7.2204 [7.0783–7.4492]","1.176","17.60"
   "jacobi/api/fused=true","dense-2048","torch-eager-out","8.4909 [8.1763–8.8082]","7.7591 [7.6395–7.8702]","1.094","9.43"
   "none/prepared/fused=false","dense-1024","numpy-out","10.2126 [10.0784–10.4172]","9.9686 [9.5383–10.2351]","1.024","2.45"
   "none/prepared/fused=false","dense-2048","numpy-out","140.2050 [138.7118–141.5010]","109.5718 [94.8426–119.9772]","1.280","27.96"
   "none/prepared/fused=false","dense-2048","torch-eager-out","140.2050 [138.7118–141.5010]","120.7513 [118.7305–123.6690]","1.161","16.11"
   "none/api/fused=false","dense-1024","numpy-out","8.6828 [8.6804–8.8068]","8.6116 [8.4711–8.8094]","1.008","0.83"
   "none/api/fused=false","dense-2048","numpy-out","131.6490 [130.8589–140.3086]","111.8489 [109.4148–117.0682]","1.177","17.70"
   "none/api/fused=false","dense-2048","torch-eager-out","131.6490 [130.8589–140.3086]","116.0280 [112.9043–128.4565]","1.135","13.46"
   "jacobi/prepared/fused=false","dense-1024","numpy-out","1.1113 [1.0821–1.1378]","1.0067 [0.9510–1.1347]","1.104","10.39"
   "jacobi/prepared/fused=false","dense-2048","numpy-out","9.9477 [9.7127–9.9914]","7.5049 [7.3675–7.5702]","1.325","32.55"
   "jacobi/prepared/fused=false","dense-2048","torch-eager-out","9.9477 [9.7127–9.9914]","8.3512 [7.8973–9.1199]","1.191","19.12"
   "jacobi/api/fused=false","dense-2048","numpy-out","8.7879 [8.4863–8.8939]","7.1452 [6.8059–7.8433]","1.230","22.99"
   "jacobi/api/fused=false","dense-2048","torch-eager-out","8.7879 [8.4863–8.8939]","7.6505 [7.4301–7.8587]","1.149","14.87"

CPU BLAS was unchanged and **not remeasured in this campaign**. The :doc:`historical explicit-BLAS results from 2026-10-15 <cpu>` name their own compiled revision and environment; they do not establish a current all-CPU win or replace the native losses above.

One fixed CPU confirmation; all original observations retained
--------------------------------------------------------------

One predeclared unpreconditioned, fused API configuration was measured once more in ABBA+BAAB order: four new artifact blocks per side. The primary pooled comparison includes both original blocks plus all four new blocks per side, across all four sizes and native/NumPy/Torch. The original strict regressions above remain unchanged; this was not an adaptive repeat-until-green campaign.

``confirmation-only`` overall: **inconclusive**; 5 within 5%, 7 inconclusive, 0 regression, 0 comparison failed.

``pooled-original-plus-confirmation`` overall: **inconclusive**; 0 within 5%, 12 inconclusive, 0 regression, 0 comparison failed.

.. csv-table:: All 12 pooled rows; six blocks / 180 rounds per side
   :class: benchmark-results
   :header: "case","candidate","A ms [process range]","B ms [process range]","B/A","95% ratio interval","strict verdict","A work","B work","B true residual","threshold"

   "dense-64","native","0.0385 [0.0362–0.0405]","0.0444 [0.0374–0.0622]","1.154","[1.017, 1.392]","inconclusive","49/51","49/51","2.072e-06","2.738e-06"
   "dense-64","numpy-out","0.2079 [0.2000–0.2191]","0.2282 [0.2143–0.3111]","1.098","[1.031, 1.370]","inconclusive","49/51","49/51","2.047e-06","2.738e-06"
   "dense-64","torch-eager-out","0.3123 [0.3017–0.3193]","0.3452 [0.3045–0.5495]","1.105","[1.005, 1.542]","inconclusive","49/51","49/51","2.447e-06","2.738e-06"
   "dense-256","native","0.4177 [0.4007–0.4600]","0.4478 [0.4186–0.6153]","1.072","[0.986, 1.292]","inconclusive","67/70","67/70","5.707e-06","6.342e-06"
   "dense-256","numpy-out","0.6049 [0.5498–0.6359]","0.6641 [0.5601–0.9365]","1.098","[0.965, 1.410]","inconclusive","67/70","67/70","5.647e-06","6.342e-06"
   "dense-256","torch-eager-out","0.7793 [0.6819–0.8009]","0.8326 [0.6642–1.4588]","1.068","[0.930, 1.644]","inconclusive","67/70","67/70","5.726e-06","6.342e-06"
   "dense-1024","native","8.0291 [7.7366–8.7181]","8.1066 [7.7409–11.6967]","1.010","[0.940, 1.412]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "dense-1024","numpy-out","8.0161 [7.7588–8.3671]","8.1998 [7.7106–12.0328]","1.023","[0.963, 1.474]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "dense-1024","torch-eager-out","8.5258 [8.1452–9.0440]","8.6422 [8.1308–13.9336]","1.014","[0.948, 1.561]","inconclusive","70/73","70/73","9.505e-06","1.079e-05"
   "dense-2048","native","99.5810 [93.0948–192.7386]","97.5272 [92.5475–151.5539]","0.979","[0.745, 1.445]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "dense-2048","numpy-out","82.0481 [76.6729–170.6077]","81.9235 [76.8896–130.9053]","0.998","[0.732, 1.472]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "dense-2048","torch-eager-out","85.9732 [78.5051–177.8665]","84.4984 [80.4061–135.2453]","0.983","[0.741, 1.509]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"

CUDA launch implementation; unchanged prepared/device scopes
------------------------------------------------------------

Whole-batch strict verdict: **inconclusive**. Native rows: 21 within 5%, 3 inconclusive, 0 regression, 0 comparison failed. Every original result remains in this table; a lower point estimate alone is not a passed strict gate.

``torch-cuda-eager`` controls: 7 within 5%, 9 inconclusive, 0 regression, 0 comparison failed.

.. csv-table:: Every native A/B result
   :class: benchmark-results
   :header: "configuration","case","candidate","A ms [process range]","B ms [process range]","B/A","95% ratio interval","strict verdict","A iterations/matvecs","B iterations/matvecs","B true residual","threshold"

   "none/prepared/fused=true/device","dense-64","cuda-native","5.6423 [5.6304–5.6787]","4.3399 [4.1682–4.4648]","0.769","[0.699, 0.886]","within 5%","49/51","49/51","2.045e-06","2.738e-06"
   "none/prepared/fused=true/device","dense-64","cuda-cublas","5.6457 [5.6221–5.7783]","4.7677 [4.7024–5.1161]","0.844","[0.765, 0.960]","within 5%","49/51","49/51","2.206e-06","2.738e-06"
   "none/prepared/fused=true/device","dense-256","cuda-native","7.3222 [7.2508–7.4000]","6.7549 [6.7068–6.7730]","0.923","[0.894, 0.949]","within 5%","67/70","67/70","5.993e-06","6.342e-06"
   "none/prepared/fused=true/device","dense-256","cuda-cublas","7.5940 [7.3911–7.8866]","6.8987 [6.6540–7.0047]","0.908","[0.818, 0.955]","within 5%","67/70","67/70","5.916e-06","6.342e-06"
   "none/prepared/fused=true/device","dense-1024","cuda-native","9.3523 [9.3125–9.3873]","9.0912 [9.0306–9.2297]","0.972","[0.954, 0.999]","within 5%","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=true/device","dense-1024","cuda-cublas","8.4293 [8.3157–8.5335]","8.3815 [8.3127–8.4177]","0.994","[0.964, 1.021]","within 5%","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=true/device","dense-2048","cuda-native","17.3020 [17.1161–17.6760]","17.3392 [17.3218–17.4454]","1.002","[0.974, 1.027]","within 5%","70/73","70/73","1.323e-05","1.474e-05"
   "none/prepared/fused=true/device","dense-2048","cuda-cublas","15.6735 [15.2615–16.0154]","15.9033 [15.8273–16.0726]","1.015","[0.970, 1.061]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "jacobi/prepared/fused=true/device","dense-64","cuda-native","0.7544 [0.7313–0.7853]","0.6754 [0.6671–0.6892]","0.895","[0.821, 0.966]","within 5%","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=true/device","dense-64","cuda-cublas","0.8000 [0.7844–0.8017]","0.7580 [0.7580–0.7619]","0.947","[0.877, 1.057]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=true/device","dense-256","cuda-native","0.6880 [0.6682–0.7069]","0.6392 [0.6391–0.6409]","0.929","[0.875, 0.988]","within 5%","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=true/device","dense-256","cuda-cublas","0.7506 [0.7268–0.7831]","0.6762 [0.6673–0.6909]","0.901","[0.826, 0.973]","within 5%","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=true/device","dense-1024","cuda-native","0.9659 [0.9495–0.9812]","0.9085 [0.9071–0.9141]","0.941","[0.909, 0.985]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=true/device","dense-1024","cuda-cublas","0.8619 [0.8619–0.8653]","0.7976 [0.7897–0.8019]","0.925","[0.876, 0.982]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=true/device","dense-2048","cuda-native","1.2842 [1.2497–1.3012]","1.2212 [1.2129–1.2280]","0.951","[0.906, 0.994]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "jacobi/prepared/fused=true/device","dense-2048","cuda-cublas","1.1438 [1.1276–1.1794]","1.1102 [1.0969–1.1191]","0.971","[0.919, 1.045]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "none/prepared/fused=true/device","poisson-8","cuda-native","3.0758 [2.6722–3.8269]","2.6011 [2.6011–2.6014]","0.846","[0.646, 1.031]","within 5%","26/28","26/28","2.012e-07","3.055e-07"
   "none/prepared/fused=true/device","poisson-32","cuda-native","12.8803 [12.6022–14.4773]","12.3506 [11.7441–12.6582]","0.959","[0.789, 1.087]","inconclusive","129/133","129/133","1.328e-06","1.367e-06"
   "none/prepared/fused=true/device","poisson-64","cuda-native","30.8441 [29.7714–31.6823]","27.6389 [27.4885–27.6922]","0.896","[0.847, 0.957]","within 5%","290/297","290/297","2.773e-06","2.810e-06"
   "none/prepared/fused=true/device","poisson-128","cuda-native","40.2580 [39.2600–41.3931]","38.9227 [38.3924–39.5243]","0.967","[0.916, 1.013]","within 5%","395/404","395/404","5.661e-06","5.749e-06"
   "jacobi/prepared/fused=true/device","poisson-8","cuda-native","3.0939 [3.0837–3.2870]","2.6334 [2.5714–2.7711]","0.851","[0.719, 0.952]","within 5%","26/28","26/28","2.012e-07","3.055e-07"
   "jacobi/prepared/fused=true/device","poisson-32","cuda-native","16.0244 [15.9783–16.2272]","13.2916 [12.6663–14.4427]","0.829","[0.733, 0.991]","within 5%","129/133","129/133","1.328e-06","1.367e-06"
   "jacobi/prepared/fused=true/device","poisson-64","cuda-native","34.3289 [34.1409–34.5150]","31.6657 [31.2515–32.0494]","0.922","[0.891, 0.956]","within 5%","290/297","290/297","2.773e-06","2.810e-06"
   "jacobi/prepared/fused=true/device","poisson-128","cuda-native","44.8943 [44.0848–46.0453]","42.6465 [42.6465–42.6703]","0.950","[0.908, 0.985]","within 5%","395/404","395/404","5.661e-06","5.749e-06"

.. csv-table:: All framework controls
   :class: benchmark-results
   :header: "configuration","case","candidate","A ms [process range]","B ms [process range]","B/A","95% ratio interval","strict verdict","A iterations/matvecs","B iterations/matvecs","B true residual","threshold"

   "none/prepared/fused=true/device","dense-64","torch-cuda-eager","13.9748 [12.9417–14.8139]","12.6274 [12.6211–12.6605]","0.904","[0.815, 0.998]","within 5%","49/51","49/51","2.206e-06","2.738e-06"
   "none/prepared/fused=true/device","dense-256","torch-cuda-eager","17.2267 [17.0871–17.3182]","17.1953 [16.9474–17.2825]","0.998","[0.967, 1.027]","within 5%","67/70","67/70","5.651e-06","6.342e-06"
   "none/prepared/fused=true/device","dense-1024","torch-cuda-eager","17.4671 [17.4511–17.5644]","17.8851 [17.7589–18.1346]","1.024","[1.004, 1.044]","within 5%","70/73","70/73","9.505e-06","1.079e-05"
   "none/prepared/fused=true/device","dense-2048","torch-cuda-eager","24.7398 [24.0704–26.0747]","25.7085 [25.6352–25.8954]","1.039","[0.971, 1.079]","inconclusive","70/73","70/73","1.323e-05","1.474e-05"
   "jacobi/prepared/fused=true/device","dense-64","torch-cuda-eager","1.9667 [1.9002–2.0517]","1.8911 [1.8866–1.8972]","0.962","[0.885, 1.075]","inconclusive","5/7","5/7","1.141e-07","2.738e-06"
   "jacobi/prepared/fused=true/device","dense-256","torch-cuda-eager","1.6377 [1.5824–1.7293]","1.5830 [1.5645–1.5957]","0.967","[0.872, 1.056]","inconclusive","4/6","4/6","1.099e-06","6.342e-06"
   "jacobi/prepared/fused=true/device","dense-1024","torch-cuda-eager","1.7921 [1.6220–1.8222]","1.5904 [1.5891–1.6058]","0.887","[0.813, 1.027]","within 5%","4/6","4/6","1.722e-07","1.079e-05"
   "jacobi/prepared/fused=true/device","dense-2048","torch-cuda-eager","1.8595 [1.7699–1.9978]","1.7353 [1.7353–1.7559]","0.933","[0.817, 1.040]","within 5%","3/5","3/5","1.245e-05","1.474e-05"
   "none/prepared/fused=true/device","poisson-8","torch-cuda-eager","7.4530 [6.5675–8.1146]","7.5784 [7.4172–7.6237]","1.017","[0.854, 1.255]","inconclusive","26/28","26/28","2.012e-07","3.055e-07"
   "none/prepared/fused=true/device","poisson-32","torch-cuda-eager","31.1815 [30.3226–32.7041]","36.2604 [34.5338–37.6677]","1.163","[1.045, 1.252]","inconclusive","129/133","129/133","1.328e-06","1.367e-06"
   "none/prepared/fused=true/device","poisson-64","torch-cuda-eager","72.6044 [68.7849–82.4225]","75.7878 [73.0645–77.6817]","1.044","[0.879, 1.135]","inconclusive","290/297","290/297","2.773e-06","2.810e-06"
   "none/prepared/fused=true/device","poisson-128","torch-cuda-eager","95.3272 [94.4449–96.8430]","103.8424 [101.8308–105.5142]","1.089","[1.010, 1.178]","inconclusive","395/404","395/404","5.661e-06","5.749e-06"
   "jacobi/prepared/fused=true/device","poisson-8","torch-cuda-eager","9.3422 [8.8440–10.2706]","8.8418 [8.2586–9.6400]","0.946","[0.775, 1.164]","inconclusive","26/28","26/28","2.012e-07","3.055e-07"
   "jacobi/prepared/fused=true/device","poisson-32","torch-cuda-eager","45.5204 [43.5759–47.6907]","43.5702 [42.2954–45.1413]","0.957","[0.882, 1.038]","within 5%","129/133","129/133","1.328e-06","1.367e-06"
   "jacobi/prepared/fused=true/device","poisson-64","torch-cuda-eager","101.8374 [100.2783–112.3567]","105.1070 [103.4498–109.1952]","1.032","[0.905, 1.106]","inconclusive","290/297","290/297","2.773e-06","2.810e-06"
   "jacobi/prepared/fused=true/device","poisson-128","torch-cuda-eager","133.7335 [133.7335–133.8111]","133.4945 [132.7613–136.5448]","0.998","[0.982, 1.034]","within 5%","395/404","395/404","5.661e-06","5.749e-06"

Changed same-candidate work rows: **0**; no per-iteration normalization.

CUDA fresh API, explicit Session reuse and prepared plans
---------------------------------------------------------

These are resident-input completed-wall measurements. Fresh API includes a new output, workspace/handle/scratch lifecycle and per-call Jacobi. Session reuses only iterative workspace/resources, allocates a fresh output and recomputes Jacobi per call; it caches neither the operator nor M. Prepared plans cache an owning operator snapshot and M and reuse output/workspace. Torch follows the corresponding allocation contract with its own caching allocator and project-written eager CG recurrence. Its native host/device execution selector is disclosed and does not change Torch's engine.

Each cell uses two fresh processes in forward/reverse configuration passes. Values are the median of the two process medians, with their min–max range; those ranges are not confidence intervals. Scopes charge different work, so cross-scope and host/device differences are descriptive and are never strict A/B verdicts.

.. csv-table:: Descriptive same-artifact native/Torch median inventory
   :class: benchmark-results
   :header: "scope","native mode","candidate","lower native median","rows","different native/Torch work"

   "api","device","cuda-cublas","4","8","0"
   "api","device","cuda-native","16","16","0"
   "api","host","cuda-cublas","4","8","0"
   "api","host","cuda-native","16","16","0"
   "prepared","device","cuda-cublas","8","8","0"
   "prepared","device","cuda-native","16","16","0"
   "prepared","host","cuda-cublas","8","8","0"
   "prepared","host","cuda-native","16","16","0"
   "session","device","cuda-cublas","8","8","0"
   "session","device","cuda-native","16","16","0"
   "session","host","cuda-cublas","8","8","0"
   "session","host","cuda-native","16","16","0"

.. csv-table:: All api native rows; native/Torch ratio below 1 means a lower native median
   :class: benchmark-results
   :header: "suite","M","mode","case","native candidate","native ms [process range]","Torch ms [process range]","native/Torch","native work","Torch work","native true residual","Torch true residual","threshold"

   "cuda-dense-f64-v1","none","host","dense-64","cuda-native","6.8317 [6.6240–7.0394]","12.8566 [12.3754–13.3377]","0.531","49/51","49/51","2.045e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","host","dense-64","cuda-cublas","8.9091 [8.6765–9.1417]","12.8566 [12.3754–13.3377]","0.693","49/51","49/51","2.206e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","host","dense-256","cuda-native","9.3219 [9.1807–9.4630]","17.3021 [16.9480–17.6562]","0.539","67/70","67/70","5.993e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","host","dense-256","cuda-cublas","11.0772 [10.9205–11.2339]","17.3021 [16.9480–17.6562]","0.640","67/70","67/70","5.916e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","host","dense-1024","cuda-native","11.8353 [11.7847–11.8859]","17.8950 [17.7652–18.0247]","0.661","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","host","dense-1024","cuda-cublas","12.8860 [12.7964–12.9755]","17.8950 [17.7652–18.0247]","0.720","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","host","dense-2048","cuda-native","19.8583 [19.6171–20.0995]","25.6089 [25.2195–25.9982]","0.775","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","host","dense-2048","cuda-cublas","20.4723 [19.9465–20.9982]","25.6089 [25.2195–25.9982]","0.799","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","device","dense-64","cuda-native","4.3666 [3.9918–4.7414]","12.4976 [12.2556–12.7395]","0.349","49/51","49/51","2.045e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","device","dense-64","cuda-cublas","6.8429 [6.6823–7.0034]","12.4976 [12.2556–12.7395]","0.548","49/51","49/51","2.206e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","device","dense-256","cuda-native","6.8552 [6.7778–6.9326]","18.8295 [17.9964–19.6627]","0.364","67/70","67/70","5.993e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","device","dense-256","cuda-cublas","9.2182 [9.2176–9.2189]","18.8295 [17.9964–19.6627]","0.490","67/70","67/70","5.916e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","device","dense-1024","cuda-native","9.4237 [9.2222–9.6253]","17.7847 [17.4812–18.0882]","0.530","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","device","dense-1024","cuda-cublas","10.3744 [10.1999–10.5489]","17.7847 [17.4812–18.0882]","0.583","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","device","dense-2048","cuda-native","17.3648 [17.1128–17.6168]","25.2308 [25.2025–25.2591]","0.688","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","device","dense-2048","cuda-cublas","18.0229 [17.8412–18.2045]","25.2308 [25.2025–25.2591]","0.714","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-64","cuda-native","1.1782 [1.1564–1.2000]","2.5677 [2.4499–2.6855]","0.459","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-64","cuda-cublas","3.0670 [3.0107–3.1233]","2.5677 [2.4499–2.6855]","1.194","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-256","cuda-native","1.0196 [1.0028–1.0364]","2.0143 [1.9079–2.1206]","0.506","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-256","cuda-cublas","2.9675 [2.9133–3.0216]","2.0143 [1.9079–2.1206]","1.473","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-1024","cuda-native","1.2840 [1.2595–1.3085]","2.2908 [2.2025–2.3792]","0.560","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-1024","cuda-cublas","3.1502 [2.9970–3.3034]","2.2908 [2.2025–2.3792]","1.375","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-2048","cuda-native","1.5889 [1.5173–1.6604]","2.1715 [2.1256–2.2174]","0.732","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-2048","cuda-cublas","3.0547 [2.9548–3.1547]","2.1715 [2.1256–2.2174]","1.407","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-64","cuda-native","0.7616 [0.7389–0.7843]","2.3936 [2.3283–2.4590]","0.318","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-64","cuda-cublas","2.6093 [2.5200–2.6986]","2.3936 [2.3283–2.4590]","1.090","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-256","cuda-native","0.7471 [0.7234–0.7707]","2.0369 [1.9728–2.1010]","0.367","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-256","cuda-cublas","2.6720 [2.6441–2.6998]","2.0369 [1.9728–2.1010]","1.312","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-1024","cuda-native","1.0086 [0.9733–1.0440]","2.2537 [2.0287–2.4787]","0.448","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-1024","cuda-cublas","2.7173 [2.5266–2.9079]","2.2537 [2.0287–2.4787]","1.206","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-2048","cuda-native","1.3276 [1.3063–1.3488]","2.2230 [2.1870–2.2590]","0.597","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-2048","cuda-cublas","2.9176 [2.8424–2.9929]","2.2230 [2.1870–2.2590]","1.312","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-csr-f64-v1","none","host","poisson-8","cuda-native","3.1153 [3.0455–3.1851]","6.1137 [5.9483–6.2791]","0.510","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","none","host","poisson-32","cuda-native","18.0034 [17.8652–18.1417]","32.3880 [31.4625–33.3135]","0.556","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","none","host","poisson-64","cuda-native","38.0655 [37.5857–38.5452]","73.0056 [72.0612–73.9500]","0.521","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","none","host","poisson-128","cuda-native","54.1683 [53.9058–54.4309]","96.0924 [96.0586–96.1262]","0.564","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","none","device","poisson-8","cuda-native","2.3929 [2.2283–2.5575]","6.4026 [5.8504–6.9548]","0.374","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","none","device","poisson-32","cuda-native","11.0328 [10.1745–11.8912]","32.5490 [30.5717–34.5263]","0.339","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","none","device","poisson-64","cuda-native","25.9250 [25.8256–26.0244]","71.3727 [69.1815–73.5639]","0.363","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","none","device","poisson-128","cuda-native","35.5576 [35.1683–35.9469]","96.2196 [94.5242–97.9149]","0.370","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-8","cuda-native","4.8662 [4.7789–4.9535]","8.9841 [8.8931–9.0751]","0.542","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","jacobi","host","poisson-32","cuda-native","24.8968 [24.8184–24.9753]","45.3477 [44.9525–45.7429]","0.549","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-64","cuda-native","56.1891 [55.8722–56.5060]","101.9474 [100.9209–102.9738]","0.551","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-128","cuda-native","79.1395 [78.3049–79.9741]","136.4509 [133.4917–139.4100]","0.580","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-8","cuda-native","2.7127 [2.6226–2.8029]","9.0698 [8.5426–9.5969]","0.299","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","jacobi","device","poisson-32","cuda-native","14.6082 [14.2953–14.9212]","49.7788 [47.2998–52.2578]","0.293","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-64","cuda-native","30.3176 [29.5307–31.1045]","100.0418 [97.9815–102.1021]","0.303","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-128","cuda-native","42.1653 [40.7842–43.5465]","132.5107 [131.3209–133.7005]","0.318","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"

.. csv-table:: All session native rows; native/Torch ratio below 1 means a lower native median
   :class: benchmark-results
   :header: "suite","M","mode","case","native candidate","native ms [process range]","Torch ms [process range]","native/Torch","native work","Torch work","native true residual","Torch true residual","threshold"

   "cuda-dense-f64-v1","none","host","dense-64","cuda-native","6.4409 [6.2005–6.6813]","12.9228 [12.4931–13.3526]","0.498","49/51","49/51","2.045e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","host","dense-64","cuda-cublas","6.8715 [6.6620–7.0810]","12.9228 [12.4931–13.3526]","0.532","49/51","49/51","2.206e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","host","dense-256","cuda-native","9.2630 [9.2324–9.2935]","17.2045 [16.7177–17.6913]","0.538","67/70","67/70","5.993e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","host","dense-256","cuda-cublas","9.1076 [9.0866–9.1287]","17.2045 [16.7177–17.6913]","0.529","67/70","67/70","5.916e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","host","dense-1024","cuda-native","12.2839 [11.9368–12.6311]","18.6189 [17.5533–19.6845]","0.660","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","host","dense-1024","cuda-cublas","11.0794 [10.6400–11.5188]","18.6189 [17.5533–19.6845]","0.595","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","host","dense-2048","cuda-native","19.7945 [19.6290–19.9599]","25.1646 [24.9305–25.3987]","0.787","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","host","dense-2048","cuda-cublas","18.5692 [18.4049–18.7336]","25.1646 [24.9305–25.3987]","0.738","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","device","dense-64","cuda-native","4.4184 [3.9722–4.8645]","12.6469 [12.3479–12.9458]","0.349","49/51","49/51","2.045e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","device","dense-64","cuda-cublas","4.9247 [4.8633–4.9861]","12.6469 [12.3479–12.9458]","0.389","49/51","49/51","2.206e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","device","dense-256","cuda-native","6.6677 [6.2935–7.0419]","17.3292 [17.1379–17.5205]","0.385","67/70","67/70","5.993e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","device","dense-256","cuda-cublas","7.3667 [6.9940–7.7394]","17.3292 [17.1379–17.5205]","0.425","67/70","67/70","5.916e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","device","dense-1024","cuda-native","9.5091 [9.4184–9.5998]","18.2552 [17.9356–18.5747]","0.521","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","device","dense-1024","cuda-cublas","8.8078 [8.7250–8.8906]","18.2552 [17.9356–18.5747]","0.482","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","device","dense-2048","cuda-native","17.3885 [17.2662–17.5109]","25.5453 [25.3888–25.7019]","0.681","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","device","dense-2048","cuda-cublas","16.1557 [16.0732–16.2381]","25.5453 [25.3888–25.7019]","0.632","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-64","cuda-native","1.1015 [1.0751–1.1278]","2.4124 [2.3807–2.4440]","0.457","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-64","cuda-cublas","1.1535 [1.0911–1.2160]","2.4124 [2.3807–2.4440]","0.478","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-256","cuda-native","1.0062 [1.0010–1.0113]","2.1549 [2.0227–2.2871]","0.467","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-256","cuda-cublas","0.9832 [0.9694–0.9971]","2.1549 [2.0227–2.2871]","0.456","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-1024","cuda-native","1.3550 [1.3094–1.4006]","2.1750 [2.0980–2.2519]","0.623","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-1024","cuda-cublas","1.2267 [1.2114–1.2420]","2.1750 [2.0980–2.2519]","0.564","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-2048","cuda-native","1.6103 [1.5925–1.6281]","2.3077 [2.2112–2.4041]","0.698","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-2048","cuda-cublas","1.4380 [1.3926–1.4835]","2.3077 [2.2112–2.4041]","0.623","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-64","cuda-native","0.8027 [0.7629–0.8425]","2.2998 [2.1794–2.4201]","0.349","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-64","cuda-cublas","0.8598 [0.8466–0.8729]","2.2998 [2.1794–2.4201]","0.374","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-256","cuda-native","0.7642 [0.7449–0.7834]","2.0341 [2.0110–2.0572]","0.376","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-256","cuda-cublas","0.7844 [0.7717–0.7971]","2.0341 [2.0110–2.0572]","0.386","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-1024","cuda-native","1.0651 [1.0558–1.0743]","2.1836 [2.1224–2.2448]","0.488","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-1024","cuda-cublas","0.9391 [0.9198–0.9583]","2.1836 [2.1224–2.2448]","0.430","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-2048","cuda-native","1.3490 [1.3417–1.3563]","2.2248 [2.1149–2.3347]","0.606","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-2048","cuda-cublas","1.2367 [1.2078–1.2655]","2.2248 [2.1149–2.3347]","0.556","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-csr-f64-v1","none","host","poisson-8","cuda-native","3.3751 [3.2767–3.4735]","6.8638 [6.8021–6.9255]","0.492","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","none","host","poisson-32","cuda-native","17.0735 [16.5414–17.6056]","29.6516 [28.4126–30.8906]","0.576","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","none","host","poisson-64","cuda-native","38.0268 [37.7248–38.3288]","72.2389 [71.9787–72.4990]","0.526","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","none","host","poisson-128","cuda-native","54.0047 [53.3658–54.6436]","100.3586 [96.7029–104.0142]","0.538","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","none","device","poisson-8","cuda-native","2.3345 [2.2651–2.4040]","6.2671 [6.0955–6.4386]","0.373","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","none","device","poisson-32","cuda-native","10.5627 [10.4353–10.6901]","31.1188 [30.8150–31.4226]","0.339","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","none","device","poisson-64","cuda-native","25.7859 [25.5527–26.0192]","69.7124 [68.7437–70.6812]","0.370","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","none","device","poisson-128","cuda-native","35.5474 [34.3324–36.7623]","94.5728 [93.5048–95.6409]","0.376","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-8","cuda-native","4.6942 [4.5808–4.8076]","9.2137 [9.0813–9.3462]","0.509","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","jacobi","host","poisson-32","cuda-native","24.7660 [24.5695–24.9625]","45.8044 [44.4537–47.1551]","0.541","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-64","cuda-native","56.1782 [54.3404–58.0160]","102.3066 [98.0306–106.5826]","0.549","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-128","cuda-native","81.1679 [78.9226–83.4132]","144.2063 [138.6464–149.7662]","0.563","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-8","cuda-native","2.8922 [2.7450–3.0394]","10.5379 [8.4859–12.5899]","0.274","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","jacobi","device","poisson-32","cuda-native","14.8867 [14.7195–15.0539]","47.7860 [46.7500–48.8220]","0.312","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-64","cuda-native","34.0727 [32.1407–36.0046]","111.7147 [104.2449–119.1845]","0.305","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-128","cuda-native","45.2147 [40.9902–49.4391]","145.6135 [134.4092–156.8178]","0.311","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"

.. csv-table:: All prepared native rows; native/Torch ratio below 1 means a lower native median
   :class: benchmark-results
   :header: "suite","M","mode","case","native candidate","native ms [process range]","Torch ms [process range]","native/Torch","native work","Torch work","native true residual","Torch true residual","threshold"

   "cuda-dense-f64-v1","none","host","dense-64","cuda-native","6.6642 [6.5761–6.7523]","13.0512 [12.5074–13.5949]","0.511","49/51","49/51","2.045e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","host","dense-64","cuda-cublas","6.8374 [6.6113–7.0636]","13.0512 [12.5074–13.5949]","0.524","49/51","49/51","2.206e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","host","dense-256","cuda-native","9.4742 [9.1256–9.8227]","18.7622 [16.9894–20.5349]","0.505","67/70","67/70","5.993e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","host","dense-256","cuda-cublas","9.3612 [9.0134–9.7090]","18.7622 [16.9894–20.5349]","0.499","67/70","67/70","5.916e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","host","dense-1024","cuda-native","11.9429 [11.6990–12.1868]","18.2972 [17.8519–18.7426]","0.653","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","host","dense-1024","cuda-cublas","10.7081 [10.6041–10.8120]","18.2972 [17.8519–18.7426]","0.585","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","host","dense-2048","cuda-native","19.7652 [19.5814–19.9490]","25.4825 [25.4728–25.4923]","0.776","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","host","dense-2048","cuda-cublas","18.2722 [18.2109–18.3335]","25.4825 [25.4728–25.4923]","0.717","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","device","dense-64","cuda-native","3.9866 [3.8742–4.0990]","12.7281 [12.3809–13.0753]","0.313","49/51","49/51","2.045e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","device","dense-64","cuda-cublas","4.9011 [4.3832–5.4189]","12.7281 [12.3809–13.0753]","0.385","49/51","49/51","2.206e-06","2.206e-06","2.738e-06"
   "cuda-dense-f64-v1","none","device","dense-256","cuda-native","6.8289 [6.7420–6.9159]","17.9631 [17.5579–18.3684]","0.380","67/70","67/70","5.993e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","device","dense-256","cuda-cublas","7.1670 [7.0710–7.2630]","17.9631 [17.5579–18.3684]","0.399","67/70","67/70","5.916e-06","5.651e-06","6.342e-06"
   "cuda-dense-f64-v1","none","device","dense-1024","cuda-native","9.4275 [9.2828–9.5722]","17.9883 [17.5610–18.4157]","0.524","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","device","dense-1024","cuda-cublas","8.4932 [8.3483–8.6381]","17.9883 [17.5610–18.4157]","0.472","70/73","70/73","9.505e-06","9.505e-06","1.079e-05"
   "cuda-dense-f64-v1","none","device","dense-2048","cuda-native","17.1752 [17.0219–17.3285]","25.5949 [25.5549–25.6349]","0.671","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","none","device","dense-2048","cuda-cublas","15.8856 [15.6489–16.1222]","25.5949 [25.5549–25.6349]","0.621","70/73","70/73","1.323e-05","1.323e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-64","cuda-native","1.0022 [0.9655–1.0389]","1.9897 [1.9648–2.0146]","0.504","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-64","cuda-cublas","0.9923 [0.9783–1.0063]","1.9897 [1.9648–2.0146]","0.499","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-256","cuda-native","0.9412 [0.9057–0.9767]","1.8176 [1.6105–2.0246]","0.518","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-256","cuda-cublas","0.9759 [0.9211–1.0307]","1.8176 [1.6105–2.0246]","0.537","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","host","dense-1024","cuda-native","1.3118 [1.2929–1.3308]","1.9978 [1.7732–2.2224]","0.657","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-1024","cuda-cublas","1.1540 [1.1077–1.2003]","1.9978 [1.7732–2.2224]","0.578","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-2048","cuda-native","1.5912 [1.5754–1.6069]","1.9501 [1.8519–2.0483]","0.816","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","host","dense-2048","cuda-cublas","1.4282 [1.3612–1.4953]","1.9501 [1.8519–2.0483]","0.732","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-64","cuda-native","0.7311 [0.7137–0.7485]","1.9476 [1.8765–2.0187]","0.375","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-64","cuda-cublas","0.7556 [0.7226–0.7886]","1.9476 [1.8765–2.0187]","0.388","5/7","5/7","1.141e-07","1.141e-07","2.738e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-256","cuda-native","0.6700 [0.6554–0.6847]","1.6443 [1.6069–1.6817]","0.407","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-256","cuda-cublas","0.6565 [0.6456–0.6674]","1.6443 [1.6069–1.6817]","0.399","4/6","4/6","1.099e-06","1.099e-06","6.342e-06"
   "cuda-dense-f64-v1","jacobi","device","dense-1024","cuda-native","0.9274 [0.9234–0.9313]","1.6576 [1.6412–1.6740]","0.559","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-1024","cuda-cublas","0.8244 [0.8157–0.8331]","1.6576 [1.6412–1.6740]","0.497","4/6","4/6","1.722e-07","1.722e-07","1.079e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-2048","cuda-native","1.2606 [1.2522–1.2690]","1.8898 [1.8479–1.9317]","0.667","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-dense-f64-v1","jacobi","device","dense-2048","cuda-cublas","1.1380 [1.1245–1.1514]","1.8898 [1.8479–1.9317]","0.602","3/5","3/5","1.245e-05","1.245e-05","1.474e-05"
   "cuda-csr-f64-v1","none","host","poisson-8","cuda-native","3.2242 [3.1873–3.2610]","6.1342 [5.6740–6.5945]","0.526","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","none","host","poisson-32","cuda-native","18.0640 [17.7476–18.3805]","32.2566 [31.0885–33.4247]","0.560","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","none","host","poisson-64","cuda-native","37.0925 [37.0580–37.1271]","69.9784 [68.6766–71.2803]","0.530","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","none","host","poisson-128","cuda-native","53.7910 [52.8938–54.6882]","100.4247 [96.1026–104.7467]","0.536","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","none","device","poisson-8","cuda-native","2.1858 [2.1755–2.1962]","5.9554 [5.9455–5.9654]","0.367","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","none","device","poisson-32","cuda-native","11.1798 [11.0599–11.2998]","32.2885 [31.3239–33.2530]","0.346","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","none","device","poisson-64","cuda-native","26.1224 [25.6408–26.6040]","71.5496 [71.0574–72.0418]","0.365","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","none","device","poisson-128","cuda-native","35.7485 [34.1550–37.3421]","94.5589 [93.4998–95.6179]","0.378","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-8","cuda-native","5.2566 [4.8006–5.7126]","10.1798 [8.9234–11.4361]","0.516","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","jacobi","host","poisson-32","cuda-native","27.5662 [25.2630–29.8694]","50.7118 [45.5096–55.9141]","0.544","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-64","cuda-native","59.4428 [57.5221–61.3636]","109.9064 [108.4558–111.3570]","0.541","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","jacobi","host","poisson-128","cuda-native","80.1842 [78.7656–81.6027]","142.7977 [138.8050–146.7904]","0.562","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-8","cuda-native","2.7074 [2.6891–2.7256]","9.0181 [8.5652–9.4709]","0.300","26/28","26/28","2.012e-07","2.012e-07","3.055e-07"
   "cuda-csr-f64-v1","jacobi","device","poisson-32","cuda-native","12.9054 [12.6392–13.1717]","44.7771 [44.3602–45.1941]","0.288","129/133","129/133","1.328e-06","1.328e-06","1.367e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-64","cuda-native","31.0486 [30.3585–31.7388]","105.9231 [99.5819–112.2643]","0.293","290/297","290/297","2.773e-06","2.773e-06","2.810e-06"
   "cuda-csr-f64-v1","jacobi","device","poisson-128","cuda-native","42.0661 [41.0829–43.0493]","134.6804 [132.4652–136.8956]","0.312","395/404","395/404","5.661e-06","5.661e-06","5.749e-06"

.. csv-table:: Every remaining native median at or above its same-artifact Torch reference
   :class: benchmark-results
   :header: "configuration","case","candidate","native ms","Torch ms","higher native median %"

   "jacobi/api/fused=true/host","dense-64","cuda-cublas","3.0670","2.5677","19.45"
   "jacobi/api/fused=true/host","dense-256","cuda-cublas","2.9675","2.0143","47.32"
   "jacobi/api/fused=true/host","dense-1024","cuda-cublas","3.1502","2.2908","37.51"
   "jacobi/api/fused=true/host","dense-2048","cuda-cublas","3.0547","2.1715","40.67"
   "jacobi/api/fused=true/device","dense-64","cuda-cublas","2.6093","2.3936","9.01"
   "jacobi/api/fused=true/device","dense-256","cuda-cublas","2.6720","2.0369","31.18"
   "jacobi/api/fused=true/device","dense-1024","cuda-cublas","2.7173","2.2537","20.57"
   "jacobi/api/fused=true/device","dense-2048","cuda-cublas","2.9176","2.2230","31.25"

All **8** remaining losses are retained. Within-configuration work gate: **true**. Fresh-pass work changes: **0**; cross-scope/mode work changes: **0**. Differences in iterations/matvecs permit a time-to-accepted-tolerance interpretation, not a pure-kernel speed comparison.

Lifecycle and launch traces are separate diagnostics
----------------------------------------------------

All profiles are instrumented single-process completed-wall samples, with three warmups and ten captured repetitions. Their before/after independent true-residual gates pass. Counts below are totals across ten repetitions; resource-only probe work counts describe gated solves outside that probe. They are not framework scores, uninstrumented speed estimates, additive phase timings or fractions of end-to-end latency. CUDA API durations can include waiting for earlier GPU work and pageable-memory handling.

Workspace captures include construction and destruction; prepare captures include both the operator snapshot and Jacobi setup. Session construction is outside timed solves. Provider-internal allocations/synchronization remain observed costs. The old pinned host scalar allocation was already absent from this campaign's baseline.

.. csv-table:: Baseline lifecycle captures
   :class: benchmark-results
   :header: "capture","compiled source","instrumented median ms","malloc/free","memcpy/stream-sync/device-sync","kernel launch API calls","gated iterations/matvecs","true residual","threshold"

   "cuda-cublas-dense-64-jacobi-api-device-solve","6bfb7b4905e7","5.198334","90/100","120/140/30","770","5/7","1.141e-07","2.738e-06"
   "cuda-cublas-dense-256-jacobi-api-device-solve","6bfb7b4905e7","3.917138","90/100","110/130/30","640","4/6","1.099e-06","6.342e-06"
   "cuda-cublas-dense-1024-jacobi-api-device-solve","6bfb7b4905e7","4.496294","90/100","110/130/30","640","4/6","1.722e-07","1.079e-05"
   "cuda-cublas-dense-2048-jacobi-api-device-solve","6bfb7b4905e7","4.661090","90/100","100/120/30","510","3/5","1.245e-05","1.474e-05"
   "cuda-cublas-dense-1024-jacobi-api-host-solve","6bfb7b4905e7","4.115286","80/90","180/200/30","400","4/6","1.722e-07","1.079e-05"
   "cuda-cublas-dense-1024-jacobi-api-host-workspace","6bfb7b4905e7","1.720404","50/60","0/20/30","0","4/6","1.722e-07","1.079e-05"
   "cuda-cublas-dense-1024-jacobi-api-device-workspace","6bfb7b4905e7","1.752028","60/70","0/20/30","0","4/6","1.722e-07","1.079e-05"
   "cuda-native-dense-1024-jacobi-api-device-solve","6bfb7b4905e7","1.568563","60/60","110/130/0","580","4/6","1.722e-07","1.079e-05"
   "torch-cuda-eager-dense-1024-jacobi-api-device-solve","6bfb7b4905e7","3.193666","0/0","210/190/20","720","4/6","1.722e-07","1.079e-05"
   "cuda-cublas-dense-1024-jacobi-api-device-prepare","6bfb7b4905e7","0.929155","30/30","10/40/0","10","4/6","1.722e-07","1.079e-05"

.. csv-table:: Baseline lifecycle captures retained identities
   :class: benchmark-results
   :header: "capture","JSON SHA256","CUDA API CSV SHA256"

   "cuda-cublas-dense-64-jacobi-api-device-solve","9539f4e5f54c38a39c77783d5c148edf7ff120ccf41818d5317cdacfd08b7a59","b7891b3fa67808f1b657291f550eae347e8dca1c2129e8b04ba5f8f39b71f9a0"
   "cuda-cublas-dense-256-jacobi-api-device-solve","b3df9a92dd67aafe30b19f8e4a23a4125398980ed00f2d8ce34e7544d99d5881","f19e36007139b58441d50574e6701defd7c44537ccaf40a6040e2180fe56d5ba"
   "cuda-cublas-dense-1024-jacobi-api-device-solve","6229dd3527f944ce8a7c0737e14a0431797c54f314ebb71afc757149a8d3b853","89cb42497fbf931dfcd21db4cdcb200482afd29f0de12ea4515c2bd5d0df1620"
   "cuda-cublas-dense-2048-jacobi-api-device-solve","7c547b2f8fd1274a286bb2d61e00140889840c419dba2fbd9f7dfff4a866d2bb","f9fadf60c3f00232b43ef1657232b8efdce85cc7ac90bb7ac34adc720c3f6b7a"
   "cuda-cublas-dense-1024-jacobi-api-host-solve","487f3e53891556b63801f75c26d3afa73b6acd1a419835f3635ff49e9974f71f","a6416c42e529f7d526ff014d82eeebc04c561d9dc8fe9e41a03caea614896956"
   "cuda-cublas-dense-1024-jacobi-api-host-workspace","0ed8ffff099cf9896e413eba0ff85e33cf0ad79592346f32137d396f4d419775","0e7d0112c3b179a23a267df0751bc6be824979502f515b323022dafdde13f148"
   "cuda-cublas-dense-1024-jacobi-api-device-workspace","7bdfc50ffd9e21df4e33fec521cd650e48137bd611611b615a9c197a866955e3","9d479726a424d67dc9298a3512df84d3ba7511b5bd700c3d2b47ae1c9beb2fca"
   "cuda-native-dense-1024-jacobi-api-device-solve","d605644afdcb5bbde0e39ba80b0352a439fb0971ac1acc6bc2dcce0d8025549d","e450a062ef597c8e64fa673b5d620f5b7d30d791f81fed04fee66ae05b63965a"
   "torch-cuda-eager-dense-1024-jacobi-api-device-solve","e303c6d6c25faee39cfc8d7244070d8596f4b6f4450ef7f6862bd8a778746204","4d56979de51d8b4170cc1b532036f43f54d8d6101b5aedbbf00e66a3e18a782d"
   "cuda-cublas-dense-1024-jacobi-api-device-prepare","48d0297b228671ff5076671aada6efeceb7f5958da81548549d8bb65e1d58ccd","6f647bb485105e8d6ad1f4095fcf9639a560cf9a4483afb9857dee0c90d6754c"

.. csv-table:: Resource/session/launch follow-up captures
   :class: benchmark-results
   :header: "capture","compiled source","instrumented median ms","malloc/free","memcpy/stream-sync/device-sync","kernel launch API calls","gated iterations/matvecs","true residual","threshold"

   "session-cuda-cublas-dense-1024-api-host-jacobi-standard","03379431a8cc","3.471923","70/80","180/200/30","400","4/6","1.722e-07","1.079e-05"
   "session-cuda-cublas-dense-1024-api-device-jacobi-standard","03379431a8cc","3.068955","80/90","110/130/30","640","4/6","1.722e-07","1.079e-05"
   "session-cuda-native-dense-1024-session-device-jacobi-standard","03379431a8cc","1.294158","20/20","110/130/0","580","4/6","1.722e-07","1.079e-05"
   "session-cuda-cublas-dense-1024-session-device-jacobi-standard","03379431a8cc","1.826763","20/20","110/130/0","640","4/6","1.722e-07","1.079e-05"
   "session-torch-cuda-eager-dense-1024-session-device-jacobi-standard","03379431a8cc","2.411116","0/0","210/190/20","720","4/6","1.722e-07","1.079e-05"
   "session-cuda-native-dense-1024-prepared-device-jacobi-standard","03379431a8cc","1.176872","0/0","100/120/0","570","4/6","1.722e-07","1.079e-05"
   "session-cuda-cublas-dense-1024-prepared-device-jacobi-standard","03379431a8cc","1.088162","0/0","100/120/0","630","4/6","1.722e-07","1.079e-05"
   "session-cuda-native-poisson-32-prepared-device-jacobi-standard","03379431a8cc","20.610954","0/0","1430/1410/0","15670","129/133","1.328e-06","1.367e-06"
   "session-cuda-native-dense-1024-prepared-device-none-explicit","03379431a8cc","23.009121","0/0","1410/1440/0","10410","69/139","8.845e-06","1.079e-05"
   "launch-cuda-cublas-dense-1024-api-device-jacobi-standard","ca0d6afebeda","2.909371","80/90","110/130/30","560","4/6","1.722e-07","1.079e-05"
   "launch-cuda-cublas-dense-1024-session-device-jacobi-standard","ca0d6afebeda","1.038850","20/20","110/130/0","560","4/6","1.722e-07","1.079e-05"
   "launch-cuda-cublas-dense-1024-prepared-device-jacobi-standard","ca0d6afebeda","1.147453","0/0","100/120/0","550","4/6","1.722e-07","1.079e-05"
   "launch-cuda-native-dense-1024-prepared-device-jacobi-standard","ca0d6afebeda","1.329903","0/0","100/120/0","490","4/6","1.722e-07","1.079e-05"
   "launch-cuda-native-poisson-32-prepared-device-jacobi-standard","ca0d6afebeda","17.301841","0/0","1430/1410/0","13090","129/133","1.328e-06","1.367e-06"
   "launch-cuda-native-dense-1024-prepared-device-none-explicit","ca0d6afebeda","21.359759","0/0","1410/1440/0","9030","69/139","8.845e-06","1.079e-05"

.. csv-table:: Resource/session/launch follow-up captures retained identities
   :class: benchmark-results
   :header: "capture","JSON SHA256","CUDA API CSV SHA256"

   "session-cuda-cublas-dense-1024-api-host-jacobi-standard","e0c5ed29fc37c418b0212ad1d097c584ed80053cac87175657cc394d4a1d3e76","3061d6c3ea95f9672d0339fe8660fbeae0c7d1a95557442b0ba2f3d8dbd2891d"
   "session-cuda-cublas-dense-1024-api-device-jacobi-standard","6e4d42ba06463ac96490dae5712150592af1927e8dfcfad8efa83ed69b9a7415","6cb2a0a20856484a1c1441641fa644ec6fef093761b8d32e9692c739ba8ed162"
   "session-cuda-native-dense-1024-session-device-jacobi-standard","9397b5c74325551495e677ab35cb29ea8c3bf8533d10134c46d0e09232636311","f6abf7df3b8a8bf96757903a578f7eff0cc08f2b10d133bb99e956c3748f4400"
   "session-cuda-cublas-dense-1024-session-device-jacobi-standard","009b0f835db32de37eb92061974d6c2622e25459fe62250b0709e3c1c4f8aad0","a5924d352c5b1c8679281bbaf00b8ac08926a1964b56d475da5a217aa77a6928"
   "session-torch-cuda-eager-dense-1024-session-device-jacobi-standard","8e502d3766a39aae5c0a3b5b01d2e7828122b844f857b02505d5b6aa2b7baf70","d8b3f8ceb6f4dbe36ebd2a50716f1f41be12c0be2abb704d9f610beb9a3802cb"
   "session-cuda-native-dense-1024-prepared-device-jacobi-standard","d09896749c1ceae71da77ef868c5f05f6cf070af2b786a3c7f13eef166f0d725","b42551c2a785da0977a7e8d98167d7f483d8fe0f9898acb6967149f0ae277561"
   "session-cuda-cublas-dense-1024-prepared-device-jacobi-standard","f4f12182a9054384c3a00e98333f6da5c275035c7f21ecab24ecf005ad66b22b","9e301d3ce64301589cb091f4c9265b52d9770189682fe25d5ef2594d39b4445b"
   "session-cuda-native-poisson-32-prepared-device-jacobi-standard","b1fa65627add53d22c0eedb42d785bffd6f1c610a2989f5c0e7dbab6771e4478","a44d161c5371031e15897d345fad1cd02cb0a19d5749527cad3940e135b8455d"
   "session-cuda-native-dense-1024-prepared-device-none-explicit","7b0f7f501d4222e1be83b633f0d9d210d276a63eb651c35e4f6379ea0dbcb90a","29ae09af64d8d8eb138feb6b322114a11df85d6acab3f722252c23c492d3a753"
   "launch-cuda-cublas-dense-1024-api-device-jacobi-standard","576229965be9b3c26d5ec18afdcbbd0930c50726ca7d475e8d9070ca2ec54a9e","f359fb47776084e439e64e094e225e633852def7992d761cf333ab84889d95ee"
   "launch-cuda-cublas-dense-1024-session-device-jacobi-standard","a884433e519f36c791d4756b09ae67703d8d725c471a37a172def6696b850b1f","9867199c82440871395de7731041150a77a7781a9e90cdac60d8f4a48c3c2a44"
   "launch-cuda-cublas-dense-1024-prepared-device-jacobi-standard","2a080b01bae7644fe27e7488b5af0fb7f87fd18972dc71a3db79f596fa30aa6f","cadbebb26e57a81c5515cbf22e1c8e5bca511d3876162c01f4e15b4fbcb9196d"
   "launch-cuda-native-dense-1024-prepared-device-jacobi-standard","9c448be520c7d776fe28e7532d115fa36a362a39836812ad7295fbca17d57148","5d7b362669db514c44407c4fd2bcf7758e26e3f1ec744b255d590caec827a356"
   "launch-cuda-native-poisson-32-prepared-device-jacobi-standard","e88eb741e1cd3b3d254a6ca3696005a47741bfad2f3da7e1e537fedf91324402","9bffa0e18134072ca9ad3ceaa5171c59b4bfc2f223ffafd4b5e1d02a54ee5fee"
   "launch-cuda-native-dense-1024-prepared-device-none-explicit","7e967d05af3dcea160311c19ddd7304313594055a53892330ddcd6e9e9cdaa97","1c2e86eaa93f38e3e19a7d6224d425804319359427d5451d2cb24be27ff2f063"

Compiled sources and retained evidence
--------------------------------------

All raw JSON, import sidecars, manifests, strict reports, frozen binaries and traces are local ignored evidence under ``build/execution-optimization-20261016/``. They are not packaged or downloadable from these documentation links; a clean clone must rerun the public commands. Stage SHA256, actually loaded extension identities and input hashes are audited independently of the live collector checkout. Later rebases and CUDA lambda whitespace fixes do not replace the compiled identities below. No claim of identical whole trees or newly measured merge heads is made.

.. csv-table:: Frozen source, stage and active binary identities
   :class: benchmark-results
   :header: "stage identity","compiled revision","identity SHA256","active module","binary SHA256"

   "build/execution-optimization-20261016/baseline-stages/cpu-release/identity.json","6bfb7b4905e79381a3d09772016aab7a860a0dad","17bc5a92fc367ab043af3aa0635156292b132d65e2beeeda67aface3d0bfb44b","_core","3111f94040afdea2519c5e61023336b8a7ef869c77bf84d7d6c4974eee8cbbff"
   "build/execution-optimization-20261016/cpu-decoupled-stage/identity.json","7a7e6ea6e998a193514a1bb060d3cc67c0aaac45","4737c72f5f80f2cf87f4383a9bafebc162c493c90e93c2f0b327c7c7d019a9ab","_core","2e01054f7cbeff105546a5149374820b5a6a5ed1886840f750a8a431fcd3f678"
   "build/execution-optimization-20261016/launch-stage/identity.json","ca0d6afebedaef1a1c4c7a5a2d2b4ffded6e1e2d","82ec5410a6d6763de6241b9002428d0e218f9b1261c74c99a6431d4c81fdc239","_core","2a6443b7d57a37eccb7a8436d8bd85618aadf4b867f3256025f4eafcefd6d2fa"
   "build/execution-optimization-20261016/launch-stage/identity.json","ca0d6afebedaef1a1c4c7a5a2d2b4ffded6e1e2d","82ec5410a6d6763de6241b9002428d0e218f9b1261c74c99a6431d4c81fdc239","_cuda","754123ef7cf27270520b5c6a687a46c6b0645f7c5d595c8750efa9e3e363ddfd"
   "build/execution-optimization-20261016/session-stage/identity.json","03379431a8cc4426447f1d5dfd011d908d9976bb","4fb7c4447f9502015a218f96c21fd5949b7d989a4c2a29916b36b2b043a6ac67","_core","2a6443b7d57a37eccb7a8436d8bd85618aadf4b867f3256025f4eafcefd6d2fa"
   "build/execution-optimization-20261016/session-stage/identity.json","03379431a8cc4426447f1d5dfd011d908d9976bb","4fb7c4447f9502015a218f96c21fd5949b7d989a4c2a29916b36b2b043a6ac67","_cuda","819cb2911a36bb5cef4055741e371d21eb3386124612c8686106cd891ba74e46"

.. csv-table:: Campaign manifest identities
   :class: benchmark-results
   :header: "campaign","path","manifest SHA256","fresh artifacts","eligible candidate/case results"

   "CPU original ABBA","build/execution-optimization-20261016/cpu-decoupled-abba/manifest.json","5d2eb61c469c5a08ca98197ddc10281be8bbedf6f603e4c6badb47495f802a5a","32","384"
   "CPU fixed confirmation","build/execution-optimization-20261016/cpu-none-api-confirmation/manifest.json","1fb60ac0d72e13d79f278f06144477dad8d3a24c219f7c4a42ee6cd6016917a8","8","96"
   "CUDA three-scope characterization","build/execution-optimization-20261016/cuda-final-scopes/manifest.json","e033557af1d083adc50d17b80b3d74679213c009f7e06656c81f876f21170bde","48","480"
   "CUDA launch ABBA","build/execution-optimization-20261016/launch-abba/manifest.json","42d8de7a2c6d229969046748ad5301669834f6473709dd292a0aa6dcc2f317dc","16","160"

Baseline lifecycle captures manifest: ``build/execution-optimization-20261016/profiles-baseline/manifest.json``, SHA256 ``130cebc1bc7e53427408638054fbefb5fb955c92e998321ee4538ac596d73e5d``.

.. csv-table:: Baseline lifecycle captures profiler program and campaign runner are different files
   :class: benchmark-results
   :header: "role","recorded path","SHA256"

   "profile_driver","scripts/profile_solve.py","e76030f99362e75913ee25b5aa0e0a6de640d073848f30c8f39000c6ae6cff2b"
   "runner","build/execution-optimization-20261016/profile_baseline.py","c0c327660a7f23ce8e3fa565c6b45ee5691b315eed589a579eb58c58d806a8b0"

retained runner SHA checked now; not recorded in historical manifest.

Resource/session/launch follow-up captures manifest: ``build/execution-optimization-20261016/profiles-followup/manifest.json``, SHA256 ``1fdfbdca06a06fbec113ba42c046f888ca224f4abb7ab099123ec390e2e2c04c``.

.. csv-table:: Resource/session/launch follow-up captures profiler program and campaign runner are different files
   :class: benchmark-results
   :header: "role","recorded path","SHA256"

   "profile_driver","scripts/profile_solve.py","a3015a157af11f2af630f937f34efc480cc3398b17c5c25e1b7b4059ae824641"
   "runner","build/execution-optimization-20261016/profile_followup.py","32d958e1b6b5c779169934058a3151389328fe262fef62efd2abd30f5c430a62"

.. csv-table:: Saved strict report identities; original reports remain intact
   :class: benchmark-results
   :header: "campaign","configuration/scope","path","SHA256","overall verdict"

   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-none-prepared","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-none-prepared.json","b5b3a1d6e9c8ab43a7ba8ba63f9341735e2afed066fc11f999e847a99f7a348a","inconclusive"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-none-api","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-none-api.json","ef3253f6d846b684398c2fba2c504e1a255fe9140b594143bb9149ac2a1a0de8","regression"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-jacobi-prepared.json","17e2af458308413a5ef9e16eff0fdf8329eca3b32db46e67616bfb9a601b9c2e","inconclusive"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-jacobi-api.json","d0f66c497d6a36aea092b2ef90b43c29441fce01df86ed16bf570410e5307b76","inconclusive"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-unfused","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-none-prepared-unfused.json","dcdb9e26351c2897675238ddccf0d87c1f5134ccbe740adf5eb46a1aa6228a6b","inconclusive"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-none-api-unfused","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-none-api-unfused.json","16f92929558f4ebb7576608a0e6b2f85f90af7130ebc76a1d7dc8d954c6e308b","inconclusive"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused.json","3c2737cbf661c29f8373ef782d8ddcf6e2bf741861399e4f9a6878cf9fee9b31","inconclusive"
   "CPU original ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","build/execution-optimization-20261016/cpu-decoupled-abba/reports/cpu-release-cpu-dense-f64-v1-jacobi-api-unfused.json","4e3b64228c444636599223a82047d1589bb95f968c27a7d137c9b27980db5b97","inconclusive"
   "CPU fixed confirmation","confirmation-only","build/execution-optimization-20261016/cpu-none-api-confirmation/reports/confirmation-only.json","a934452c6c7f135b636cbf4d8e62ed7b4d2f0df6714dc8aba19201effc309e83","inconclusive"
   "CPU fixed confirmation","pooled-original-plus-confirmation","build/execution-optimization-20261016/cpu-none-api-confirmation/reports/pooled-original-plus-confirmation.json","79803ef9c3e094653f50048a37b533b346188300f2c51a322a5d503d63cf6cc2","inconclusive"
   "CUDA launch ABBA","cuda-release-cuda-dense-f64-v1-none-prepared-device","build/execution-optimization-20261016/launch-abba/reports/cuda-release-cuda-dense-f64-v1-none-prepared-device.json","0c4c20004782a12bc334997bce25964a95c77a2c12054daf7adeb93eb2eb57dd","inconclusive"
   "CUDA launch ABBA","cuda-release-cuda-dense-f64-v1-jacobi-prepared-device","build/execution-optimization-20261016/launch-abba/reports/cuda-release-cuda-dense-f64-v1-jacobi-prepared-device.json","1960041f64c31e2479cdbbfb4c163452af0779eaa27b4debe24adf2d02629ebc","inconclusive"
   "CUDA launch ABBA","cuda-release-cuda-csr-f64-v1-none-prepared-device","build/execution-optimization-20261016/launch-abba/reports/cuda-release-cuda-csr-f64-v1-none-prepared-device.json","ce84ae52b38c6d8522c2f38e96a56a107a65251139658b30f34ead2e0dec9c55","inconclusive"
   "CUDA launch ABBA","cuda-release-cuda-csr-f64-v1-jacobi-prepared-device","build/execution-optimization-20261016/launch-abba/reports/cuda-release-cuda-csr-f64-v1-jacobi-prepared-device.json","1211f44bbc33e4e33b1f16bf13234dd7aa77fdb52fd16b054fc6ec556c8bde39","inconclusive"

.. csv-table:: Historical CPU comparator retained for the original artifact schema
   :class: benchmark-results
   :header: "campaign","frozen comparison source","source SHA256","original source path","matching frozen Python package"

   "CPU fixed confirmation","build/execution-optimization-20261016/frozen-cpu-comparator/compare_performance.py","956d0fc65dc463bb2edfdc800da2dbcbfe3049bc4ae6996844fc0c0c804f6124","scripts/compare_performance.py","build/execution-optimization-20261016/cpu-decoupled-stage/python"

The CPU confirmation uses the exact historical comparator source relocated to ignored storage with its matching frozen Python harness. The live comparator's later Session support does not rewrite historical data or change this campaign's 5% / 95% / 10,000-resample policy. Pooled original and new blocks were also checked together for exact hardware/compiler/runtime controls and complete one-thread library-pool inventories.

Public reproduction
-------------------

Follow :doc:`/build` and :doc:`/install` for project-local dependencies,
separate CPU/CUDA environments, the CUDA 12.8 toolkit and supported host
compiler. Build and validate each source before starting quiet measurements.
Use separate source/build directories; do not run ``make`` against the frozen
evidence stages. Public commands recreate the protocol on the current host,
not the original binary hashes or observed medians. The local orchestration
scripts are evidence tools and are not required public production APIs.

The exact compiled sources are retained by public tags. Fetch and verify them
before creating detached checkouts (``OUT`` must be a new local directory):

.. code-block:: bash

   PROJECT=$PWD
   OUT="$PROJECT/build/execution-reproduction"
   mkdir -p "$OUT"
   git fetch origin refs/tags/benchmarks/gemv-fusion-20261016:refs/tags/benchmarks/gemv-fusion-20261016
   test "$(git rev-parse refs/tags/benchmarks/gemv-fusion-20261016)" = 7a7e6ea6e998a193514a1bb060d3cc67c0aaac45
   git fetch origin refs/tags/benchmarks/cuda-session-20261016:refs/tags/benchmarks/cuda-session-20261016
   test "$(git rev-parse refs/tags/benchmarks/cuda-session-20261016)" = 03379431a8cc4426447f1d5dfd011d908d9976bb
   git fetch origin refs/tags/benchmarks/cuda-launch-20261016:refs/tags/benchmarks/cuda-launch-20261016
   test "$(git rev-parse refs/tags/benchmarks/cuda-launch-20261016)" = ca0d6afebedaef1a1c4c7a5a2d2b4ffded6e1e2d
   git worktree add --detach "$OUT/baseline-source" 6bfb7b4905e79381a3d09772016aab7a860a0dad
   git worktree add --detach "$OUT/cpu-source" 7a7e6ea6e998a193514a1bb060d3cc67c0aaac45
   git worktree add --detach "$OUT/session-source" 03379431a8cc4426447f1d5dfd011d908d9976bb
   git worktree add --detach "$OUT/launch-source" ca0d6afebedaef1a1c4c7a5a2d2b4ffded6e1e2d

CPU A/B uses baseline versus cpu-source. For every none/jacobi,
prepared/api and fused/no-fused combination below, invoke four fresh
processes in A,B,B,A order and save separate JSON files. A/B selects the
source and build directory only; the entire candidate list and options stay
the same. Build both sources with the recorded GNU 16 release configuration.
Example for one block (set ``SRC``, ``BUILD``, ``RUN`` for each invocation):

.. code-block:: bash

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR="$BUILD" \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 PYPROF_OUTPUT="$OUT/$RUN.json" \
       PYPROF_ARGS='--implementations native,numpy-out,torch-eager-out --preconditioner none --fused --warmups 3 --seed 20261015 --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'

Repeat the configuration loop with ``--preconditioner jacobi``, scope
``prepared``, and ``--no-fused``. The one fixed fused none/api confirmation
adds A,B,B,A,B,A,A,B once; its pooled comparison includes all original two
plus all four new blocks per side, with no case/candidate filtering.

CUDA characterization uses launch-source with all three scopes, both native
execution modes, both preconditioners and dense/CSR suites: 24 configurations
in forward order, then the exact reverse order, each in a fresh process.
The default host mode is explicit ``host`` on the CLI and omits the optional
device-engine field from historical specs. Builds use GNU 13 with CUDA 12.8,
sm75 and no fast-math. For example:

.. code-block:: bash

   taskset -c 0 make -C "$OUT/launch-source" pyprof PRESET=cuda-release \
       PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR="$OUT/launch-build" \
       CUDA_ROOT="$PROJECT/.deps/cuda-12.8" \
       CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-dense-jacobi-session-device-r1.json" \
       PYPROF_ARGS='--device cuda --cuda-device 0 --cuda-execution-mode device --implementations cuda-native,cuda-cublas,torch-cuda-eager --preconditioner jacobi --fused --warmups 3 --seed 20261015 --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'

Set ``PYPROF_SCOPE=api|session|prepared``, native mode ``host|device`` and
preconditioner ``none|jacobi`` separately. For CSR use
``PYPROF_SUITE=cuda-csr-f64-v1``, ``PYPROF_REPETITIONS=1`` and the candidates
``cuda-native,torch-cuda-eager``. Keep all cases, including losses. Pin
``taskset -c 0`` and all BLAS/OpenMP thread variables to one; the public
worker enforces its declared controls. Set
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and leave ``CUDA_LAUNCH_BLOCKING`` unset.

Launch A/B uses session-source versus launch-source in A,B,B,A order for
prepared/device only, both suites and preconditioners, with unchanged full
candidate lists and timing boundaries. Compare equal-spec blocks using the
public strict comparator (the process exits nonzero for an inconclusive or
regression verdict; preserve the report):

.. code-block:: bash

   PYTHONPATH="$SRC/python" "$PROJECT/.venv/bin/python" \
       "$SRC/scripts/compare_performance.py" \
       --baseline "$OUT/A1.json" "$OUT/A4.json" \
       --current "$OUT/B2.json" "$OUT/B3.json" \
       --max-regression 0.05 --bootstrap-samples 10000 --seed 20261015 \
       --output "$OUT/comparison.json"

Never pass different scopes or execution modes to that comparator. A raw
replay requires the actual raw JSON, its matching compiled source/runtime
and the same public CLI; these ignored local artifacts are not shipped in a
clean clone. Nsight Systems is optional host tooling, not a pip dependency.
Recreate diagnostic captures with ``scripts/profile_solve.py`` and the
``nsys`` flags in :doc:`api-lifecycle-20261016`; select the exact source,
implementation, case, scope, mode and preconditioner from the capture table.
Resource/prepare probes remain distinct from solve captures.

