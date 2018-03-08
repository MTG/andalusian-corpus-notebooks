#import pyqtgraph as pg
import numpy as np

#from cultures.makam.utilities import sort_dictionary

_NUM_CENTS_IN_OCTAVE = 1200.0


class Converter(object):
    @staticmethod
    def hz_to_cent(hz_track, ref_freq, min_freq=20.0):
        """--------------------------------------------------------------------
        Converts an array of Hertz values into cents.
        -----------------------------------------------------------------------
        hz_track : The 1-D array of Hertz values
        ref_freq : Reference frequency for cent conversion
        min_freq : The minimum frequency allowed (exclusive)
        --------------------------------------------------------------------"""
        # The 0 Hz values are removed, not only because they are meaningless,
        # but also logarithm of 0 is problematic.
        assert min_freq >= 0.0, 'min_freq cannot be less than 0'

        hz_track = np.array(hz_track).astype(float)

        # change values less than the min_freq to nan
        hz_track[hz_track <= min_freq] = np.nan

        return np.log2(hz_track / ref_freq) * _NUM_CENTS_IN_OCTAVE

    @staticmethod
    def cent_to_hz(cent_track, ref_freq):
        """--------------------------------------------------------------------
        Converts an array of cent values into Hertz.
        -----------------------------------------------------------------------
        cent_track  : The 1-D array of cent values
        ref_freq    : Reference frequency for cent conversion
        --------------------------------------------------------------------"""
        cent_track = np.array(cent_track).astype(float)

        return 2 ** (cent_track / _NUM_CENTS_IN_OCTAVE) * ref_freq


def compute_overall_histogram(histograms):
    """--------------------------------------------------------------------
    Plot an overall histogram obtained by the sum of the histograms provided
    as inputs. Every histogram is converted in cents. A new x-axis interval
    is created using minimum and maximum value in cents for all the
    histograms. All the values of the histograms are inserted in the new
    x-axis interpolating them from the original reference system.
    -----------------------------------------------------------------------
    histograms: array of histogram. Every histogram is defined as
    [[vals, bins], tonic]
    --------------------------------------------------------------------"""
    values = {}
    tonic = {}
    bins_in_cent = {}
    new_x_axis_in_cent = {}
    interpolated_values = {}
    curves = {}

    min_bin_value = 0
    max_bin_value = 0

    for mbid in histograms:

        # translate histogram in cent
        values[mbid] = histograms[mbid][0][0]
        tonic[mbid] = histograms[mbid][1]
        bins_in_cent[mbid] = Converter.hz_to_cent(histograms[mbid][0][1], tonic[mbid])

        # update the overall minimum and maximum bounds in cents
        if int(bins_in_cent[mbid][0]) < min_bin_value:
            min_bin_value = int(bins_in_cent[mbid][0])

        if int(bins_in_cent[mbid][-1]) > max_bin_value:
            max_bin_value = int(bins_in_cent[mbid][-1])

        # create the new x axis interval in cent for every function
        new_x_axis_in_cent[mbid] = range(int(bins_in_cent[mbid][0]), int(bins_in_cent[mbid][-1]), 1)

        # use interpolation to find all the values on the new x axis
        interpolated_values[mbid] = np.interp(new_x_axis_in_cent[mbid], bins_in_cent[mbid], values[mbid])

        # create a dictionary for the curves
        curves[mbid] = dict(zip(new_x_axis_in_cent[mbid], interpolated_values[mbid]))

    overall_x = range(min_bin_value, max_bin_value)
    overall_y = np.zeros(max_bin_value - min_bin_value)

    overall_hist = dict(zip(overall_x, overall_y))

    for mbid in curves:
        for key in curves[mbid]:
            overall_hist[key] += curves[mbid][key]

    plot_bins = sorted(overall_hist.keys())
    plot_vals = [overall_hist[key] for key in plot_bins]

    #pg.plot(plot_bins, plot_vals)
    return plot_bins, plot_vals