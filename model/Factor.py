import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


class BasicFactor(object):

    def __init__(self, name):
        self.name = name

    def data_procsess(self, **kwargs):
        pass

    def win(self, x, trim=0.2, rm=True):
        y = x.copy()
        x.dropna()
        if (trim < 0) | (trim > 0.5):
            print("trimming must be reasonable")
            exit()
        qtrim_min = x.quantile(trim)
        qtrim_mid = x.quantile(0.5)
        qtrim_max = x.quantile(1 - trim)
        if trim < 0.5:
            y[x < qtrim_min] = qtrim_min
            y[x > qtrim_max] = qtrim_max
        else:
            y[x != None] = qtrim_mid
        return y

    def stand(self, z, trim_num):
        x = self.win(z, trim_num)
        x_mean = np.nanmean(x)
        x_std = np.nanstd(x)
        y = (x - x_mean) / x_std
        return y

    def std_winsor(self, z):
        tmp_z = z.copy()
        tmp_z.dropna()
        z_std = np.std(tmp_z)
        min_std = -3 * z_std
        max_std = 3 * z_std
        z[z < min_std] = min_std
        z[z > max_std] = max_std
        z[z == None] = min_std
        return z

    def compute(self, x, nagtive=False):
        if len(x) == 0:
            return x
        sd_szie = self.stand(x, 0.05)
        if nagtive:
            sd_szie = - sd_szie
        sd_sd_size = self.std_winsor(sd_szie)
        try:
            size_cdf = ECDF(sd_sd_size)
            size_cdf_ = size_cdf(sd_sd_size)
        except:
            size_cdf_ = None
            print(x)
            exit()
        return size_cdf_


class SizeFactor(BasicFactor):

    def __init__(self, name):
        super(SizeFactor).__init__(name)

    def data_procsess(self, x, nagtive=True):
        return self.compute(x, nagtive)


class VolatilityFactor(BasicFactor):

    def __init__(self, name):
        super(VolatilityFactor).__init__(name)

    def data_procsess(self, x, nagtive=True):
        return self.compute(x, nagtive)


class RsiFactor(BasicFactor):

    def __init__(self, name):
        super(RsiFactor).__init__(name)

    def data_procsess(self, x, nagtive=True):
        return self.compute(x, nagtive)


class MomentaFactor(BasicFactor):
    def __init__(self, name):
        super(MomentaFactor).__init__(name)

    def data_procsess(self, x, nagtive=False):
        return self.compute(x, nagtive)


class QualityFactor(BasicFactor):

    def __init__(self, name):
        super(QualityFactor).__init__(name)

    def data_procsess(self, raw_file1):
        sd_roa = self.stand(raw_file1['Rev_Over_mktCap'], 0.05)  # 资产回报
        roa_cdf = ECDF(sd_roa)
        sd_acc = self.stand(raw_file1["q_opincome"], 0.05)  # accural ？现金流
        acc_cdf = ECDF(sd_acc)
        sd_nocfod = self.stand(raw_file1['NOCF_Over_Debt'], 0.05)
        nocfod_cdf = ECDF(sd_nocfod)
        raw_file1["quality_factor"] = 0.25 * roa_cdf(sd_roa) + 0.25 * acc_cdf(sd_acc) + 0.5 * nocfod_cdf(sd_nocfod)


class ValueFactor(BasicFactor):
    def __init__(self, name):
        super(ValueFactor).__init__(name)

    def data_procsess(self, raw_file1):
        sd_cashval = self.stand(raw_file1['Cash_Over_MktCap'], 0.05)  # 现金除以市值
        sd_sd_cashval = self.std_winsor(sd_cashval)
        cashval_cdf = ECDF(sd_sd_cashval)
        sd_roa = self.stand(raw_file1['Rev_Over_mktCap'], 0.05)  # 收益除以市值
        sd_sd_roa = self.std_winsor(sd_roa)
        roa_cdf = ECDF(sd_sd_roa)
        sd_bp = self.stand(raw_file1['pb'], 0.05)  # 市净率
        sd_sd_bp = self.std_winsor(sd_bp)
        bp_cdf = ECDF(sd_sd_bp)
        raw_file1["value_factor"] = 1 / 3 * cashval_cdf(sd_sd_cashval) \
                                    + 1 / 3 * roa_cdf(sd_sd_roa) \
                                    + 1 / 3 * bp_cdf(sd_sd_bp)