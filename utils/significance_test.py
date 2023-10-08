# significance test
import scipy.stats as stats

# two-tailed paired t-test
def paired_ttest(a, b):
    result = stats.ttest_rel(a, b)
    return result

# two-tailed independent t-test
def ind_ttest(a, b):
    result = stats.ttest_ind(a, b)
    return result

# two-tailed one sample test
def onesample_ttest(a, popmean):
    result = stats.ttest_1samp(a=a, popmean=popmean)
    return result


if __name__ == "__main__":
    a = [0.1, 0.3, 0.15, 0.6, 0.22]
    b = [0.51, 0.61, 0.21, 0.84, 0.30]
    print(paired_ttest(a, b))
    print(onesample_ttest(a, 0.01))