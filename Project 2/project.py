# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


clean_title = lambda title: title if pd.isnull(title) else ('registered nurse' if title.strip().lower() == 'rn' else title.strip().lower())

def clean_loans(loans):
    cleaned_loans = loans.copy()
    cleaned_loans['issue_d'] = pd.to_datetime(cleaned_loans['issue_d'], format='%b-%Y')
    cleaned_loans['term'] = cleaned_loans['term'].apply(lambda t: int(t.split()[0]))
    cleaned_loans['emp_title'] = cleaned_loans['emp_title'].apply(clean_title)
    cleaned_loans['term_end'] = cleaned_loans.apply(
        lambda row: row['issue_d'] + pd.DateOffset(months=row['term']),
        axis=1
    )
    return cleaned_loans


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def correlations(df, pairs):
    result = {}
    for col1, col2 in pairs:
        r = df[col1].corr(df[col2])
        result[f"r_{col1}_{col2}"] = r
    return pd.Series(result)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    bins   = [580, 670, 740, 800, 850]
    labels = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']

    df = loans.copy()
    df['term'] = df['term'].astype(str).str.extract(r'(\d+)', expand=False)
    
    
    # Cut fico_range_low into our 4 bins, then cast to string
    df['fico_bin'] = (
        pd.cut(
            df['fico_range_low'],
            bins=bins,
            labels=labels,
            right=False,
            ordered=True
        )
        .astype(str)
    )

    # Boxplot
    fig = px.box(
        df,
        x='fico_bin',
        y='int_rate',
        color='term',
        category_orders={
            'fico_bin': labels,
            'term': ['36', '60']
        },
        color_discrete_map={
            '36': 'purple',
            '60': 'gold'
        },
    )

    # Finalize layout—axes, title, legend title
    fig.update_layout(
        title='Interest Rate vs. Credit Score',
        xaxis_title='Credit Score Range',
        yaxis_title='Interest Rate (%)',
        legend_title_text='Loan Length (Months)'
    )

    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    df = loans.copy()
    df['has_ps'] = df['desc'].notna()

    observed_diff = df[df['has_ps']]['int_rate'].mean() - df[~df['has_ps']]['int_rate'].mean()

    diffs = []
    for _ in range(N):
        shuffled = np.random.permutation(df['has_ps'])
        df['shuffled_ps'] = shuffled
        sim_diff = df[df['shuffled_ps']]['int_rate'].mean() - df[~df['shuffled_ps']]['int_rate'].mean()
        diffs.append(sim_diff)

    return (np.array(diffs) >= observed_diff).mean()

    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    return "Borrowers might skip personal statement because of hidden reasones like financial struggles, which can also impact interest rates."


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    tax_sum = 0
    for i in range(len(brackets)):
        tax_rate = brackets[i][0]
        lower_bound = brackets[i][1]
        upper_bound = brackets[i + 1][1] if i < len(brackets) - 1 else float('inf')

        taxable_amt = min(income, upper_bound) - lower_bound
        if taxable_amt <= 0:
            continue

        tax_sum += tax_rate * taxable_amt

        if income <= upper_bound:
            break

    return tax_sum


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw):
    df = state_taxes_raw.dropna(how='all').copy()   # drop fully blank separator rows
    mask_foot = df['State'].astype(str).str.startswith('(')
    df.loc[mask_foot, 'State'] = pd.NA
    df['State'] = df['State'].ffill()
    df = df[df['Rate'].notna()].reset_index(drop=True)

    # 'none' --> 0% brackets
    df['Rate'] = (
        df['Rate']
          .str.lower()
          .replace('none', '0%')
          .fillna('0%')           # in case NaN
          .str.rstrip('%')
          .astype(float)
          .div(100)
          .round(2)
    )

    # Bracket limits: fill missing with "$0", strip "$" & commas, to int
    df['Lower Limit'] = (
        df['Lower Limit']
          .fillna('$0')
          .astype(str)
          .str.replace(r'[\$,]', '', regex=True)
          .astype(int)
    )

    # Keep only 3 columns
    df = df[['State', 'Rate', 'Lower Limit']]

    return df


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    grouped = state_taxes.groupby('State', sort=False)
    bracket_series = grouped.apply(
        lambda df: list(
            zip(
                df.sort_values('Lower Limit')['Rate'],
                df.sort_values('Lower Limit')['Lower Limit']
            )
        )
    )
    return bracket_series.to_frame(name='bracket_list')


def combine_loans_and_state_taxes(loans, state_taxes):
    import json
    brackets_df = state_brackets(state_taxes).reset_index()

    mapping_path = Path('data') / 'state_mapping.json'
    with open(mapping_path, 'r') as f:
        state_mapping = json.load(f)

    # Map tax‐table names → USPS codes
    brackets_df['State'] = brackets_df['State'].map(state_mapping)

    # Rename loans column and merge
    df = loans.rename(columns={'addr_state': 'State'})
    return df.merge(brackets_df, on='State', how='left')


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
        (0.10, 0),
        (0.12, 11000),
        (0.22, 44725),
        (0.24, 95375),
        (0.32, 182100),
        (0.35, 231251),
        (0.37, 578125)
    ]

    df = loans_with_state_taxes.copy()

    df['federal_tax_owed'] = df['annual_inc'].apply(lambda income: tax_owed(income, FEDERAL_BRACKETS))
    df['state_tax_owed'] = df.apply(lambda row: tax_owed(row['annual_inc'], row['bracket_list']), axis=1)
    df['disposable_income'] = df['annual_inc'] - df['federal_tax_owed'] - df['state_tax_owed']

    expected_cols = list(loans_with_state_taxes.columns) + ['federal_tax_owed', 'state_tax_owed', 'disposable_income']
    return df[expected_cols]


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    results = []

    for keyword in keywords:
        mask_series = loans['emp_title'].str.contains(keyword, na=False)

        # Filter 
        filtered_loans = loans[mask_series]

        # Mean by category
        mean_series = filtered_loans.groupby(categorical_column)[quantitative_column].mean()
        grouped_df = mean_series.to_frame()

        # Rename 
        new_col_name = f'{keyword}_mean_{quantitative_column}'
        grouped_df.columns = [new_col_name]
        
        overall_mean_value = filtered_loans[quantitative_column].mean()
        grouped_df.loc['Overall'] = overall_mean_value

        results.append(grouped_df)

    # Combine
    combined_df = pd.concat(results, axis=1)

    # Sorted 
    index_list = list(combined_df.index)
    sorted_index = sorted(index_list, key=lambda x: (x == 'Overall', x))

    final_df = combined_df.reindex(sorted_index)
    return final_df


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    df = aggregate_and_combine(loans, keywords,
                               quantitative_column,
                               categorical_column)
    
    # subgroups: all means in one direction?
    sub_gt = (df.iloc[:-1, 0] > df.iloc[:-1, 1]).all()
    sub_lt = (df.iloc[:-1, 0] < df.iloc[:-1, 1]).all()
    
    # aggregate: overall mean flips direction?
    agg_gt = df.iloc[-1, 0] > df.iloc[-1, 1]
    agg_lt = df.iloc[-1, 0] < df.iloc[-1, 1]

    return bool((sub_gt and agg_lt) or (sub_lt and agg_gt))

def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': ['teacher', 'manager'],
        'quantitative_column': 'loan_amnt',
        'categorical_column': 'verification_status'
    }