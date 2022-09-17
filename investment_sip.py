# SIP calculator
import matplotlib.pyplot as plt
import pandas as pd

# Fixed components
start_sip = 30000
monthly_sip = start_sip
total_years = 30
year = [i + 1 for i in range(total_years)]

# Variables
roi_list = [0.12, 0.15, 0.2, 0.25]
yearly_raise_list = [0, 0.1]

# Initialization
total_fund = monthly_sip
money_invested = monthly_sip
total_fund_at_each_year = {}
total_fund_combination = {}
yearly_sip_amount = {}
monthly_total_fund = {}


def get_total_invested(monthly_sip_, yearly_raise_, total_years):
    total_invested_ = monthly_sip_
    for month_ in range(1, 12 * total_years):
        if month_ % 12 == 0:
            monthly_sip_ = monthly_sip_ * (1 + yearly_raise_)
        total_invested_ = total_invested_ + monthly_sip_
    return total_invested_


# Calculation
for roi in roi_list:
    print(f"{roi * 100}%")
    for yearly_raise in yearly_raise_list:
        total_fund = start_sip
        monthly_sip = start_sip
        money_invested = start_sip

        total_invested = get_total_invested(start_sip, yearly_raise, total_years)
        condition_key = f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%__total_invested_Rs.{round(total_invested):,}"

        total_fund_at_each_year[condition_key] = []
        total_fund_combination[condition_key] = []
        yearly_sip_amount[condition_key] = []
        monthly_total_fund[condition_key] = []

        for month in range(1, total_years * 12):
            if month % 12 == 0:
                monthly_sip = monthly_sip * (1 + yearly_raise)
                total_fund_at_each_year[condition_key].append(total_fund)
                yearly_sip_amount[condition_key].append(monthly_sip)
            total_fund = total_fund * (1 + roi / 12) + monthly_sip
            money_invested = money_invested + monthly_sip

        total_fund_at_each_year[condition_key].append(total_fund)
        total_fund_combination[condition_key].append(total_fund)
        yearly_sip_amount[condition_key].append(monthly_sip)

        print(f"----ROI: {roi * 100}% AND Yearly_Raise: {yearly_raise * 100}%----")
        print(f"total value of investment after {total_years} years at {roi * 100}% CAGR would be {round(total_fund)}")
        print(f"Total money invested: Rs. {round(money_invested)}, with last monthly SIP = {monthly_sip}")
        print("------------------------------------------")

for combination in pd.DataFrame(total_fund_at_each_year).columns:
    plt.plot(year, total_fund_at_each_year[combination])
plt.title(f"Start SIP: Rs.{start_sip}")
plt.xlabel("Years completed")
plt.ylabel("Total fund value")
plt.legend(pd.DataFrame(total_fund_at_each_year).columns.tolist())
plt.show()
