import numpy as np
from sklearn.linear_model import LinearRegression

import time
from datetime import date, timedelta

import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline, BSpline

def moving_avg(x, y, w):
    '''
    if len(x) != len(y):
        print('x, y dimension mismatch')
        return

    if len(x) < 5:
        print('Too few values')
    '''
    x_fill = []
    y_fill = []
    for i in range(1, len(x)):
        dist = int(x[i] - x[i-1])
        for j in range(0, dist):
            x_fill.append(x[i-1, 0] + j)
            y_fill.append(y[i-1] + j * (y[i] - y[i-1]) / dist)

    x_fill.append(x[-1, 0])
    y_fill.append(y[-1])

    x_out = []
    y_out = []
    deriv_out = []

    print(x_fill)

    for i in range(w, len(x_fill) - w):
        if i in x:
            model = LinearRegression().fit(np.array(x_fill[i - w : i + w + 1]).reshape((-1, 1)), y_fill[i - w : i + w + 1])
            x_out.append(x_fill[i])
            y_out.append(model.predict(np.array([x_fill[i]]).reshape((-1, 1)))[0])
            deriv_out.append(model.coef_[0] * 7)

    return (x_out, y_out, deriv_out)

            

dates = [
    date(2020, 9, 30),
    date(2020, 10, 1), 
    date(2020, 10, 2), 
    date(2020, 10, 3),
    #date(2020, 10, 5),
    date(2020, 10, 6),
    date(2020, 10, 7),
    date(2020, 10, 8),
    date(2020, 10, 9),
    date(2020, 10, 10),
    date(2020, 10, 11),
    date(2020, 10, 12),
    date(2020, 10, 13),
    date(2020, 10, 14),
    date(2020, 10, 15),
    date(2020, 10, 16),
    date(2020, 10, 17),
    date(2020, 10, 18),
]

dates = np.array([int((x - dates[0]).days) for x in dates]).reshape((-1, 1))
weights = np.array([
    83.5,
    83.9,
    83.7,
    83.8,
    #84.8,
    82.4,
    82.7,
    82.4,
    82.4,
    81.8,
    81.9,
    82.3,
    82.0,
    81.3,
    81.2,
    80.6,
    82.0,
    80.8,
])

fm_lbm_ratio = 0.9

start_bf = 0.17
start_weight = 83.8
start_fm = start_bf * start_weight
start_lbm = start_weight - start_fm

width = 3
(dates_avg, weights_avg, rates_avg) = moving_avg(dates, weights, width)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Subplot 1
ax1.set_xticks(np.arange(0, dates[-1, 0] + 1, 1))

# Actual weight-points
ax1.plot(dates, weights, 'r.', markersize=8)
ax1.plot(dates_avg, weights_avg, 'g.', markersize=8)

# Smoothed weight-points
dates_smooth = np.linspace(dates[0], dates[-1], 300)
spl = make_interp_spline(dates[:,0].tolist(), weights, k=3)  # type: BSpline
weight_smooth = spl(dates_smooth)
ax1.plot(dates_smooth, weight_smooth, 'r--', label='Actual weight')

# Plot average weight smooth
dates_avg_smooth = np.linspace(dates_avg[0], dates_avg[-1], 300)
avg_spl = make_interp_spline(dates_avg, weights_avg, k=2)  # type: BSpline
avg_weight_smooth = avg_spl(dates_avg_smooth)
ax1.plot(dates_avg_smooth, avg_weight_smooth, 'g--', label='Moving avg weight')

ax1.set_yticks(np.arange(80.0, 85.0, 0.5))
ax1.grid()
ax1.legend()

# Subplot 2
(dates_avg2, weights_avg2, rates_avg2) = moving_avg(np.array(dates_avg).reshape((-1, 1)), np.array(weights_avg), 2)
ax2.plot([x + 2 for x in dates_avg2], [-x for x in rates_avg2], 'b.', markersize=8)  

dates_avg_smooth2 = np.linspace(dates_avg2[0], dates_avg2[-1], 300)
avg_rate_spl2 = make_interp_spline(dates_avg2, [-x for x in rates_avg2], k=2)  # type: BSpline
avg_rate_smooth2 = avg_rate_spl2(dates_avg_smooth2)
ax2.plot([x + 2 for x in dates_avg_smooth2], avg_rate_smooth2, 'b--', label='Moving weight loss rate')

#ax2.plot(dates_avg, [-x for x in rates_avg], 'b.', markersize=8)  

#avg_rate_spl = make_interp_spline(dates_avg, [-x for x in rates_avg], k=2)  # type: BSpline
#avg_rate_smooth = avg_rate_spl(dates_avg_smooth)
#ax2.plot(dates_avg_smooth, avg_rate_smooth, 'b--', label='Moving weight loss rate')

ax2.set_yticks(np.arange(0.0, 2.5, 0.25))
ax2.set_ylim([0.0, 2.5])
ax2.grid()
ax2.legend()

print("Current deriv: {}".format(rates_avg2[-1]))

ra = range(dates_avg[-1], dates[-1, 0] + 1, 1)
ax1.plot(ra, [weights_avg[-1] + rates_avg2[-1] * (x - dates_avg[-1]) / 7 for x in ra], 'gx:')

end_date = int((date(2020, 11, 30) - date(2020, 9, 30)).days)
end_weight = weights_avg2[-1] + rates_avg2[-1] * (end_date - dates_avg2[-1]) / 7
print("Weight at end date: {:.2f} kg".format(end_weight))

end_fm = start_fm - (start_weight - end_weight) * fm_lbm_ratio
end_lbm = start_lbm - (start_weight - end_weight) * (1.0 - fm_lbm_ratio)

print("End body fat percent: {:.2f}".format(100.0 * end_fm / (end_fm + end_lbm)))

plt.show()