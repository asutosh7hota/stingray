
import numpy as np

from nose.tools import raises

from stingray import Lightcurve

np.random.seed(20150907)


class TestLightcurve(object):

    def setUp(self):
        self.times = [1, 2, 3, 4]
        self.counts = [2, 2, 2, 2]
        self.dt = 1.0

    def test_create(self):
        """
        Demonstrate that we can create a trivial Lightcurve object.
        """
        lc = Lightcurve(self.times, self.counts)

    def test_lightcurve_from_toa(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)

    def test_tstart(self):
        tstart = 0.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt, tstart=0.0)
        assert lc.tstart == tstart
        assert lc.time[0] == tstart + 0.5*self.dt

    def test_tseg(self):
        tstart = 0.0
        tseg = 5.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt,
                                        tseg=tseg, tstart=tstart)

        assert lc.tseg == tseg
        assert lc.time[-1] - lc.time[0] == tseg-self.dt

    def test_nondivisble_tseg(self):
        """
        If the light curve length input is not divisible by the time resolution,
        the last (fractional) time bin will be dropped.
        """
        tstart = 0.0
        tseg = 5.5
        lc = Lightcurve.make_lightcurve(self.times, self.dt,
                                        tseg=tseg, tstart=tstart)
        assert lc.tseg == int(tseg/self.dt)

    def test_correct_timeresolution(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        assert np.isclose(lc.dt, self.dt)


    def test_bin_correctly(self):
        ncounts = np.array([2, 1, 0, 3])
        tstart = 0.0
        tseg = 4.0

        toa = np.hstack([np.random.uniform(i, i+1, size=n) for i,n \
                          in enumerate(ncounts)])

        dt = 1.0
        lc = Lightcurve.make_lightcurve(toa, dt, tseg=tseg, tstart=tstart)

        assert np.allclose(lc.counts, ncounts)

    def test_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0+dt/2.,5-dt/2., dt)
        counts = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, counts)
        assert np.allclose(lc.countrate, np.zeros_like(counts)+mean_counts/dt)


    @raises(TypeError)
    def test_init_with_none_data(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0+dt/2.,5-dt/2., dt)
        counts = np.array([None for i in xrange(times.shape[0])])
        lc = Lightcurve(times, counts)

    @raises(AssertionError)
    def test_init_with_inf_data(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0+dt/2.,5-dt/2., dt)
        counts = np.array([np.inf for i in xrange(times.shape[0])])
        lc = Lightcurve(times, counts)

    @raises(AssertionError)
    def test_init_with_nan_data(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0+dt/2.,5-dt/2., dt)
        counts = np.array([np.nan for i in xrange(times.shape[0])])
        lc = Lightcurve(times, counts)

    def test_method_works(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        lc.to_txt("test_lc.txt", use_counts=True)

    def test_method_saves_counts_correctly(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        lc.to_txt("test_lc.txt", use_counts=True)
        lc_loaded = np.loadtxt("test_lc.txt")
        assert np.allclose(lc_loaded[:,1], lc.counts)

    def test_method_saves_countrate_correctly(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        lc.to_txt("test_lc.txt", use_counts=False)
        lc_loaded = np.loadtxt("test_lc.txt")
        assert np.allclose(lc_loaded[:,1], lc.countrate)

    @raises(AssertionError)
    def save_method_needs_string_input(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        lc.to_txt([1,2,3], use_counts=False)

    def test_read_lightcurve_works(self):
        filename = "test_tte.txt"
        np.savetxt(filename, self.times)
        dt = 0.1
        lc = Lightcurve.from_txt(filename, tte=True, dt=dt)

    def test_read_tte_data_works_correctly(self):
        dt = 0.1
        filename = "test_tte.txt"
        lc = Lightcurve.from_txt(filename, tte=True, dt=dt)
        assert np.isclose(lc.dt, dt)
        assert np.isclose(lc.tseg, self.times[-1]-self.times[0])

    def test_read_binned_data(self):
        lc = Lightcurve.from_txt("test_lc.txt", tte=False)

    def test_read_data_usecols(self):
        timestamps = np.linspace(0,1000,1000)
        data = np.random.randint(0,100, size=(4,timestamps.shape[0]))
        data = np.vstack([timestamps, data])
        np.savetxt("test_usecols.txt", data)
        usecols = [0,2]
        lc = Lightcurve.from_txt("test_usecols.txt", tte=False, usecols=usecols)

    @raises(AssertionError)
    def test_read_data_usecols_fails(self):
        usecols = [0,2,4]
        lc = Lightcurve.from_txt("test_usecols.txt", tte=False, usecols=usecols)


class TestLightcurveRebin(object):

    def setUp(self):
        dt = 1.0
        n = 10
        mean_counts = 2.0
        times = np.arange(dt/2, dt/2+n*dt, dt)
        counts= np.zeros_like(times)+mean_counts
        self.lc = Lightcurve(times, counts)

    def test_rebin_even(self):
        dt_new = 2.0
        lc_binned = self.lc.rebin_lightcurve(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)
        assert np.isclose(self.lc.tseg, lc_binned.tseg)
        counts_test = np.zeros_like(lc_binned.time) + \
                      self.lc.counts[0]*dt_new/self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)


    def test_rebin_odd(self):
        dt_new = 1.5
        lc_binned = self.lc.rebin_lightcurve(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)

        counts_test = np.zeros_like(lc_binned.time) + \
                      self.lc.counts[0]*dt_new/self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)
