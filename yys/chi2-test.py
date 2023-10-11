import numpy as np
from scipy.stats import chi2_contingency

def chi2_simulation():
    # Define your list of winning probability values
    winning_probabilities = [0.55, 0.76, 0.7, 0.99, 0.61, 0.71, 0.61, 0.73, 0.61, 0.90, 0.52, 0.58, 0.95, 0.84, 0.59]
    print("Data Len:", len(winning_probabilities))

    # Number of iterations for the simulation
    iterations = 10000

    # Number of bins or intervals for the Chi-Square test
    num_bins = 2  # You can adjust the number of bins as needed

    # Significance level (alpha) for hypothesis testing
    alpha = 0.05

    # Function to perform a single Chi-Square goodness-of-fit test
    def chi_square_test(probabilities, num_bins):
        observed_frequencies, _ = np.histogram(probabilities, bins=num_bins, range=(0, 1))
        expected_frequencies = [len(probabilities) / num_bins] * num_bins

        chi2, p = chi2_contingency([observed_frequencies, expected_frequencies])[:2]

        print(f"Observed frequencies: {observed_frequencies}")
        print(f"Expected frequencies: {expected_frequencies}")
        print(f"Chi2: {chi2}, p: {p}")

        return p

    # Perform the Chi-Square test 10,000 times
    p_values = []
    for _ in range(iterations):
        # Shuffle the list of winning probabilities to simulate randomness
        np.random.shuffle(winning_probabilities)
        
        # Perform the Chi-Square test for the shuffled data and store the p-value
        p_value = chi_square_test(winning_probabilities, num_bins)
        p_values.append(p_value)

    # Count how many times the null hypothesis is rejected (p < alpha)
    reject_count = sum(1 for p in p_values if p < alpha)

    # Calculate the proportion of rejections
    proportion_rejected = reject_count / iterations

    print(f"Proportion of rejections: {proportion_rejected:.4f}")


def binomial_test():

    # Define your list of binary outcomes (1 for win, 0 for lose)
    binary_outcomes = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]

    # Define the expected probability of success (p) under a uniform distribution
    expected_p = 0.5  # Assuming equal probability for win and lose

    # Perform the binomial test
    p_value = binom_test(sum(binary_outcomes), n=len(binary_outcomes), p=expected_p, alternative='two-sided')

    # Set the significance level (alpha)
    alpha = 0.05

    # Check if the p-value is less than alpha to determine significance
    if p_value < alpha:
        print("Reject the null hypothesis: The outcomes do not follow a uniform distribution.")
    else:
        print("Fail to reject the null hypothesis: The outcomes follow a uniform distribution.")


if __name__ == "__main__":
    chi2_simulation()