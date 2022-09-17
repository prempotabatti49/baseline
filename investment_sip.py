# SIP calculator
import matplotlib.pyplot as plt
import pandas as pd

# Fixed components
start_sip = 30_000
monthly_sip = 30_000
total_years = 35
year = [i + 1 for i in range(total_years)]

# Variables
roi_list = [0.12]
yearly_raise_list = [0, 0.05]

# Initialization
total_fund = monthly_sip
money_invested = monthly_sip
total_fund_at_each_year = {}
total_fund_combination = {}
yearly_sip_amount = {}

# Calculation
for roi in roi_list:
    print(f"{roi*100}%")
    for yearly_raise in yearly_raise_list:
        total_fund = start_sip
        monthly_sip = start_sip

        total_fund_at_each_year[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"] = []
        total_fund_combination[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"] = []
        yearly_sip_amount[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"] = []

        for month in range(1, total_years * 12):
            if month % 12 == 0:
                monthly_sip = monthly_sip * (1 + yearly_raise)
                total_fund_at_each_year[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"].append(total_fund)
                yearly_sip_amount[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"].append(monthly_sip)
            total_fund = total_fund * (1 + roi / 12) + monthly_sip
            money_invested = money_invested + monthly_sip

        total_fund_at_each_year[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"].append(total_fund)
        total_fund_combination[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"].append(total_fund)
        yearly_sip_amount[f"roi_{roi * 100}%__yearly_raise_{yearly_raise * 100}%"].append(monthly_sip)

        print(f"----ROI: {roi*100}% AND Yearly_Raise: {yearly_raise*100}%----")
        print(f"total value of investment after {total_years} years at {roi * 100}% CAGR would be {round(total_fund)}")
        print(f"Total money invested: Rs. {round(money_invested)}, with last monthly SIP = {monthly_sip}")
        print("------------------------------------------")


for combination in pd.DataFrame(total_fund_at_each_year).columns:
    plt.plot(year, total_fund_at_each_year[combination])
plt.xlabel("Years completed")
plt.ylabel("Total fund value")
plt.legend(pd.DataFrame(total_fund_at_each_year).columns.tolist())
plt.show()

