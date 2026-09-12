"""
====================================================================
 LABORATORY EXERCISE - Series1.pdf
 Combined Python implementation for Exercises 1, 2, and 3
====================================================================

This single file contains all three exercises, clearly separated
into labeled sections:

    PART 1 -> (1 + 1/n)^n  approaching e, for compounding periods
              from "yearly" up to "every nanosecond"

    PART 2 -> (a^h - 1)/h  approaching ln(a), for a = 2, e, 3,
              as h shrinks from 0.1 down to 1e-7

    PART 3 -> e^x = sum_{n=0}^{inf} x^n / n!  (x = 1),
              partial sums up to N = 10,000 terms

Each part prints its own table to the console and then opens its
own Matplotlib figure (a separate window per part), so all 3
figures pop up when you run this file, one after another.

Run this file with:
    python Series1_lab.py
====================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# PART 1: (1 + 1/n)^n  ->  e
# ====================================================================
def exercise1():
    """
    Compute (1 + 1/n)^n for increasingly frequent compounding periods,
    from "yearly" all the way up to "every nanosecond".

    As n gets bigger and bigger, (1 + 1/n)^n gets closer and closer to
    the mathematical constant e = 2.718281828...
    """

    print("\n" + "=" * 60)
    print("PART 1: (1 + 1/n)^n approaching e")
    print("=" * 60)

    # ---- STEP 1: define how often compounding happens (label, n) ----
    compounding_periods = [
        ("yearly",              1),
        ("twice a year",        2),
        ("quarterly",           4),
        ("monthly",             12),
        ("weekly",              52),
        ("daily",               365),
        ("hourly",              365 * 24),
        ("every minute",        365 * 24 * 60),
        ("every second",        365 * 24 * 60 * 60),
        ("every millisecond",   365 * 24 * 60 * 60 * 1_000),
        ("every microsecond",   365 * 24 * 60 * 60 * 1_000_000),
        ("every nanosecond",    365 * 24 * 60 * 60 * 1_000_000_000),
    ]

    labels = [item[0] for item in compounding_periods]
    n_values = [item[1] for item in compounding_periods]

    # ---- STEP 2: compute (1 + 1/n)^n for every n ----
    results = []
    for n in n_values:
        # NOTE: for very large n (like every microsecond/nanosecond),
        # computing (1 + 1/n) directly loses precision because 1/n
        # becomes smaller than what a float64 can add to 1 accurately.
        # We use math.log1p(x), which accurately computes log(1 + x)
        # even when x is extremely small, then undo the log with exp().
        #   (1 + 1/n)^n  =  exp( n * log(1 + 1/n) )  =  exp( n * log1p(1/n) )
        value = math.exp(n * math.log1p(1 / n))
        results.append(value)

    e_true = np.e  # true value of e, for comparison

    # ---- STEP 3: print the table, just like in the PDF ----
    print("How often          n                     (1 + 1/n)^n")
    print("-" * 60)
    for label, n, value in zip(labels, n_values, results):
        print(f"{label:<18} {n:<20} {value:.6f}")
    print("-" * 60)
    print(f"True value of e = {e_true:.6f}")

    # ---- STEP 4: draw the histogram (bar chart) in Matplotlib ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exercise 1: Convergence of (1 + 1/n)^n to e", fontsize=13, fontweight="bold")

    x_positions = np.arange(len(labels))

    # left panel: bar chart of the values themselves
    bars = ax1.bar(x_positions, results, color="tab:blue")
    ax1.axhline(y=e_true, color="red", linestyle="--", label=f"e = {e_true:.6f}")
    ax1.set_title("Value per compounding period")
    ax1.set_ylabel("(1 + 1/n)^n")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()

    for bar, value in zip(bars, results):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{value:.6f}", ha="center", va="bottom", fontsize=7, rotation=90)

    # right panel: bar chart of the error, log scale
    errors = [abs(value - e_true) for value in results]
    ax2.bar(x_positions, errors, color="tab:orange")
    ax2.set_yscale("log")
    ax2.set_title("Error shrinks like e/(2n)")
    ax2.set_ylabel("|(1+1/n)^n - e|  (log scale)")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("exercise1_output.png", dpi=120)


# ====================================================================
# PART 2: (a^h - 1)/h  ->  ln(a)
# ====================================================================
def exercise2():
    """
    Compute (a^h - 1) / h for three different bases "a" (2, e, and 3),
    as h shrinks towards 0.

    This expression is the definition of the derivative of a^x at
    x = 0. As h gets smaller and smaller, (a^h - 1)/h settles down
    to the value of ln(a) (the natural logarithm of a).
    """

    print("\n" + "=" * 60)
    print("PART 2: (a^h - 1)/h approaching ln(a)")
    print("=" * 60)

    # ---- STEP 1: define the three bases and the shrinking h values ----
    bases = {
        "a = 2": 2,
        "a = 2.71828...": math.e,   # this is our "a = e" column
        "a = 3": 3,
    }

    h_values = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]

    # Tolerance requested in the PDF (used only to decide when we can
    # say the value has "settled").
    TOLERANCE = 1e-6

    # ---- STEP 2: compute (a^h - 1)/h for every base and every h ----
    results = {name: [] for name in bases}
    for name, a in bases.items():
        for h in h_values:
            value = (a ** h - 1) / h
            results[name].append(value)

    # the true limits are just the natural logarithms of each base
    true_limits = {name: math.log(a) for name, a in bases.items()}

    # ---- STEP 3: print the table, exactly like in the PDF ----
    header = f"{'h':<12}" + "".join(f"{name:<18}" for name in bases)
    print(header)
    print("-" * len(header))
    for i, h in enumerate(h_values):
        row = f"{h:<12}" + "".join(f"{results[name][i]:<18.4f}" for name in bases)
        print(row)
    print("-" * len(header))
    settle_row = f"{'settles at':<12}" + "".join(f"{true_limits[name]:<18.4f}" for name in bases)
    print(settle_row)
    print(f"\n(Tolerance used to judge convergence: {TOLERANCE})")

    # ---- STEP 4: draw the histogram (grouped bar chart) in Matplotlib ----
    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(h_values)          # number of h values -> number of groups
    n_bars = len(bases)                # number of bars per group (one per base)
    bar_width = 0.8 / n_bars
    group_positions = np.arange(n_groups)

    colors = ["tab:blue", "tab:green", "tab:red"]

    for i, (name, color) in enumerate(zip(bases, colors)):
        # shift each base's bars so the 3 bars in a group sit side by side
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(group_positions + offset, results[name], width=bar_width,
               label=f"{name} (ln a = {true_limits[name]:.4f})", color=color)
        # dashed horizontal line at the true limit ln(a)
        ax.axhline(y=true_limits[name], color=color, linestyle="--", linewidth=1)

    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"h = {h:g}" for h in h_values])
    ax.set_ylabel("(a^h - 1)/h")
    ax.set_title("Exercise 2: (a^h - 1)/h settling to ln(a)")
    ax.legend(title="Bars: difference quotient per h.  Dashed lines: the limit ln(a).",
              fontsize=8, title_fontsize=9)

    plt.tight_layout()
    plt.savefig("exercise2_output.png", dpi=120)


# ====================================================================
# PART 3: e^x = sum x^n / n!  (Taylor / Maclaurin series)
# ====================================================================
def exercise3():
    """
    Compute e^x using its Taylor (Maclaurin) series:

        e^x = sum_{n=0}^{infinity}  x^n / n!

    For this exercise we use x = 1, so the series becomes:

        e^1 = 1/0! + 1/1! + 1/2! + 1/3! + ...

    We add up terms one by one, up to N = 10,000 terms, and watch
    the partial sum get closer and closer to the real value of e.
    """

    print("\n" + "=" * 60)
    print("PART 3: e^x Taylor series (x = 1), up to N = 10,000 terms")
    print("=" * 60)

    # ---- STEP 1: settings ----
    x = 1                  # we are approximating e^1 = e
    N_MAX = 10_000          # add terms up to this many (as required by the PDF)

    # these are the specific N values we will show in the printed table
    # and use as the bars in the histogram (so the chart isn't cluttered
    # with 10,000 bars)
    checkpoints = [1, 2, 3, 5, 10, 20, 50, 100, 1000, N_MAX]

    # ---- STEP 2: compute the running (partial) sum, term by term ----
    partial_sum = 0.0
    term = 1.0              # this holds x^n / n! ; starts at n = 0 -> x^0/0! = 1
    checkpoint_sums = []     # sum recorded at each checkpoint N
    checkpoint_labels = []

    # We loop n = 0, 1, 2, ..., N_MAX - 1 and keep a running total.
    # Instead of recomputing x^n and n! from scratch every time (which
    # would be slow and could overflow for large n), we build each new
    # term from the previous one:
    #       term_n = term_(n-1) * x / n
    for n in range(N_MAX):
        if n > 0:
            term = term * x / n
        partial_sum += term

        step_number = n + 1  # number of terms added so far
        if step_number in checkpoints:
            checkpoint_sums.append(partial_sum)
            checkpoint_labels.append(str(step_number))

    e_true = math.e

    # ---- STEP 3: print a table of partial sums at the checkpoints ----
    print(f"{'N (terms)':<12}{'Partial sum S_N':<20}{'|S_N - e|':<15}")
    print("-" * 47)
    for label, s in zip(checkpoint_labels, checkpoint_sums):
        error = abs(s - e_true)
        print(f"{label:<12}{s:<20.10f}{error:<15.2e}")
    print("-" * 47)
    print(f"True value of e = {e_true:.10f}")

    # ---- STEP 4: draw the plots in Matplotlib ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exercise 3: e^x = sum x^n / n!  (x = 1), summation up to N = 10,000 terms",
                 fontsize=12, fontweight="bold")

    # left panel: histogram (bar chart) of partial sums
    x_positions = np.arange(len(checkpoint_labels))
    bars = ax1.bar(x_positions, checkpoint_sums, color="tab:purple")
    ax1.axhline(y=e_true, color="red", linestyle="--", label=f"e = {e_true:.6f}")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(checkpoint_labels, rotation=45, ha="right")
    ax1.set_xlabel("number of terms N in the summation")
    ax1.set_ylabel("S_N = sum, up to N terms")
    ax1.set_title("Histogram of the partial sums")
    ax1.legend()

    for bar, s in zip(bars, checkpoint_sums):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{s:.6f}", ha="center", va="bottom", fontsize=7, rotation=90)

    # right panel: correct decimal digits vs N (log-log)
    errors = [abs(s - e_true) for s in checkpoint_sums]
    # avoid log(0) if a partial sum happens to match e exactly (float rounding)
    errors = [err if err > 0 else 1e-16 for err in errors]
    n_terms_numeric = [int(label) for label in checkpoint_labels]

    ax2.plot(n_terms_numeric, errors, marker="o", color="tab:green")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("number of terms N in the summation (log scale)")
    ax2.set_ylabel("|S_N - e|  (log scale)")
    ax2.set_title("How many correct digits each N buys (log-log)")

    for n_val, err in zip(n_terms_numeric, errors):
        digits = max(0, int(-math.log10(err))) if err > 0 else 16
        ax2.annotate(f"{digits} digits", (n_val, err), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("exercise3_output.png", dpi=120)


# ====================================================================
# MAIN - run all three exercises in order
# ====================================================================
if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()

    # Show all three figures at once (one window per exercise).
    # Close the figure windows to end the program.
    plt.show()
