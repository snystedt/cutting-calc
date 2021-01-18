import time
from datetime import date, timedelta
from enum import Enum
import math
import matplotlib.pyplot as plt

class Gender(Enum):
    MALE = 1
    FEMALE = 2

def bmr_mifflin_stjeor(bw, height, age, gender):
    return 10.0 * bw + 6.25 * height - 5.0 * age + (5 if gender is Gender.MALE else -161)

def bmr_katch_mcardle(lbm):
    return 370 + 21.6 * lbm

# Sedentary 1.2
# Lightly active 1.375
# Moderately active 1.55
# Very active 1.725
# Extra active 1.9
def tdee_activity_level(bmr, multiplier):
    return bmr * multiplier

def print_stats(bw, fat_percentage, lbm, fm, bmr, tdee):
    print("Body weight:    {:7.2f}".format(bw))
    print("Fat percent:    {:7.2f}".format(fat_percentage))
    print("Lean body mass: {:7.2f}".format(lbm))
    print("Fat mass:       {:7.2f}".format(fm))
    print("BMR:            {:7.2f}".format(bmr))
    print("TDEE:           {:7.2f}".format(tdee))

def print_data(date_list, bw_list, fat_percent_list, lbm_list, fm_list):
    day = {
        0: 'Mon',
        1: 'Tue',
        2: 'Wed',
        3: 'Thu',
        4: 'Fri',
        5: 'Sat',
        6: 'Sun'
    }
    header = "| {:15s} | {:6s} | {:6s} | {:6s} | {:6s} |".format("Date", "Weight", "Fat %", "LBM", "FM")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(0, len(date_list)):
        print("| {:15s} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} |".format(day[date_list[i].weekday()] + " " + date_list[i].isoformat(), bw_list[i], fat_percent_list[i] * 100, lbm_list[i], fm_list[i]))
    print("-" * len(header))

    

def main():
    # kg
    bw = 84.0
    fat_percentage = 0.185
    fm = bw * fat_percentage
    lbm = bw - fm

    gender = Gender.MALE
    # cm
    height = 175
    # years
    age = 27

    bmr = bmr_katch_mcardle(lbm)
    tdee = tdee_activity_level(bmr, 1.55)
    print_stats(bw, fat_percentage, lbm, fm, bmr, tdee)

    deficit = -550
    lbm_fat_ratio = 0.1

    # Amount to lose in total
    lose = 9
    weeks = int(math.ceil(lose / (-deficit * 7 * 0.453 / 3500))) + 1

    # Start date
    start = date(2020, 1, 18)

    date_list = [start + x * timedelta(days=7) for x in range(0, weeks)]
    bw_list = [bw + x * (7 * deficit / (3500/0.453)) for x in range(0, weeks)]
    lbm_list = [lbm + x * lbm_fat_ratio * (7 * deficit / (3500/0.453)) for x in range(0, weeks)]
    fm_list = [fm + x * (1.0 - lbm_fat_ratio) * (7 * deficit / (3500/0.453)) for x in range(0, weeks)]
    fat_percent_list = [f / (f + l) for f, l in zip(fm_list, lbm_list)]
    print_data(date_list, bw_list, fat_percent_list, lbm_list, fm_list)


if __name__ == "__main__":
    main()