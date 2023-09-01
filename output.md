## Output

This is the output of `usage.py h5/lwfa.h5`

Data generated using PIConGPU 0.5.0.

h5/lwfa.h5 contains iteration 126175, at 13.33 ps.

Swapping y and z axes.

The particle bunch is propagating along the z direction.

The dataset contains 3,356,185 macroparticles, corresponding to 13,369,720,832 'real' electrons.

Descriptive statistics of the dataset:

```

       position_x_um  position_y_um  ...       weights    energy_mev
count   3.356185e+06   3.356185e+06  ...  3.356185e+06  3.356185e+06
mean    3.650167e+01   3.171529e+01  ...  3.983607e+03  2.085688e+02
std     6.424610e+00   1.142403e+01  ...  1.253296e+03  1.386893e+02
min     2.887242e-04   1.977612e-04  ...  1.034352e+01  5.000027e+00
25%     3.209686e+01   2.298315e+01  ...  2.535706e+03  8.172086e+01
50%     3.826371e+01   3.595817e+01  ...  3.944592e+03  2.012327e+02
75%     4.075175e+01   3.977995e+01  ...  5.282948e+03  3.144294e+02
max     7.919274e+01   7.784132e+01  ...  5.442342e+03  9.286193e+02

[8 rows x 8 columns]

```

The (weighted) mean energy is 2.485225e+02 MeV.

Wrote plots/phase_space.png

<a href="plots/phase_space.png"><img src="plots/phase_space.png" width="200"></a>

Reducing number of particles.

The dataset contains 1,680,082 macroparticles, corresponding to 1,680,082 'real' electrons.

Descriptive statistics of the dataset:

```

       position_x_um  position_y_um  ...    weights    energy_mev
count   1.680082e+06   1.680082e+06  ...  1680082.0  1.680082e+06
mean    3.728759e+01   3.349535e+01  ...        1.0  2.484482e+02
std     5.898530e+00   1.048681e+01  ...        0.0  1.340014e+02
min     2.877399e-02   1.977612e-04  ...        1.0  5.000275e+00
25%     3.431252e+01   2.681100e+01  ...        1.0  1.241444e+02
50%     3.894848e+01   3.749608e+01  ...        1.0  2.918729e+02
75%     4.090325e+01   4.021048e+01  ...        1.0  3.333022e+02
max     7.909510e+01   7.783379e+01  ...        1.0  9.286193e+02

[8 rows x 8 columns]

```

Wrote plots/comparative_phase_space.png

<a href="plots/comparative_phase_space.png"><img src="plots/comparative_phase_space.png" width="200"></a>

Writing dataframe to file. This may take a while...

Wrote h5/lwfa.txt

