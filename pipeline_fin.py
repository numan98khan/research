import argparse
import os
import shutil

import cv2
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from scipy.signal import butter
from scipy.signal import lfilter, filtfilt
from scipy.signal import argrelmin, argrelmax
import matplotlib.patches as patches
import utils
from utils import interpolate_beat
import json


# from sklearn.decomposition import FastICA, PCA
# import yaml

class VideoProcessor():
    def __init__(self, sample_rate, super_folder):
        self.sample_rate = sample_rate
        self.super_folder = super_folder
        pass

    ## Basic Filters
    def red_channel_mean(self, frames):

        start_frame = 60
        frame_gap = 1800

        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[..., 2].mean()
            
            try:
                signal.append(mean_of_r_ch - signal[-1])
            except:
                signal.append(mean_of_r_ch)
                
        signal = np.array(signal)
        signal = signal[start_frame: start_frame + frame_gap]

        return signal

    def luma_component_mean(self, frames):
        
        start_frame = 60
        frame_gap = 1800
        
        signal = []
        for frame_bgr in frames:
            img_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            mean_of_luma = img_ycrcb[..., 0].mean()
            signal.append(mean_of_luma)

        signal = np.array(signal)
        
        
        signal = signal[start_frame: start_frame + frame_gap]

        return signal

    ## advanced Filters
    def cut_start(self, signal, seconds, **kwargs):
        n_frames = self.sample_rate * seconds
        return np.concatenate((np.full(n_frames, np.nan), signal[n_frames:]), axis=0)

    def rolling_average(self, signal, **kwargs):
        window_size_seconds = kwargs["window_size_seconds"]
        window_size = int(window_size_seconds * self.sample_rate)
        if window_size % 2 == 0:
            window_size += 1
        y = np.convolve(signal, np.ones(window_size), 'valid') / window_size
        y = np.pad(y, [((window_size - 1) // 2, (window_size - 1) // 2)], mode='edge')
        return y

    def subtract(self, signal, **kwargs):
        original_signal = kwargs["prev_x"]
        assert signal.shape == original_signal.shape
        y = original_signal-signal
        return y

    def butter_lowpass_filter(self, signal, low, filter_order, **kwargs):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = low / nyq
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, signal)
        return y


    ## Video to signal processors
    def proc_ext_sig(self, file, user_fold, output_folder):
        filepath = os.path.join(user_fold, file)

        fname = file.split(".")[0]
        
        user = fname.split('/')[-1] 
        fname = fname.split('/')[-1]

        directory = os.path.join(output_folder, user)
        if not os.path.exists(directory):
            os.makedirs(directory)

        csv_fpath = os.path.join(output_folder, user, fname + ".csv")

        print(csv_fpath)

        if not os.path.isfile(csv_fpath):
            columns, extracted_s = [], []

            list_of_frames = []
            vidcap = cv2.VideoCapture(filepath)
            success, frame = vidcap.read()
            h, w, _ = frame.shape
            ioo = 0
            while success:
                list_of_frames.append(frame)
                success, frame = vidcap.read()
                ioo+=1

            vidcap.release()

            i = 0

            
            # for extractor in params["extractor"]:
            # print(extractor)

            for ext_name in ['red_mean', 'luma_mean']:

                columns.append(ext_name)
                sys.stdout.write(
                    "{} ({}/{}),{},{}\n".format(user, i + 1, len([]), file, ext_name))
                sys.stdout.flush()
                
                # f_output = None
                if ext_name == 'red_mean':
                    f_output = self.red_channel_mean(frames=list_of_frames)
                elif ext_name == 'luma_mean':
                    f_output = self.luma_component_mean(frames=list_of_frames)
                
                extracted_s.append(f_output.tolist())

                i += 1



            extracted_s = np.array(extracted_s) * -1
            # assert extracted_s.ndim == 2, "Different functions resulted in different length of extracted signal"
            df = pd.DataFrame(extracted_s.T, columns=columns)
            df.to_csv(csv_fpath, sep=",", float_format="%.4f", index=False)


            ## for visualization
            # visualize_signal(
            #     extracted_s,
            #     labels=columns,
            #     output_fname=csv_fpath.replace('.csv', '.jpeg'),)
        else:
            sys.stdout.write("Skipping, file %s exists already\n" % fname)
            sys.stdout.flush()


    def extract_signal(self, save, vid_dir):
        slist = os.listdir(vid_dir)

        for i in slist:
            # fname = "C:/Users/numan98khan/Desktop/HemaRays/ml/HemaRays_Data_OneFourth/Gender_Female_Age_7_HB_12_P9_OF.MOV"
            fname = vid_dir + i
            
            # print(i)
            

            if '.MOV' in fname:
                
                print(fname)
            
                uFold = fname.split('.mov'.upper())[0]
                oFolder = self.super_folder+'extracted/'


                # process_single_file(fname, uFold, oFolder, params, 
                #         [], Extractor(sample_rate=params["frame_rate"]))

                self.proc_ext_sig(fname, uFold, oFolder)

                # print()

        # exit()


    ## Signal to Signal Processors
    def preprocess_signal(self):
        # sp = Preprocessor(sample_rate=frame_rate)

        signals_directory = self.super_folder+"extracted/"
        output_folder = self.super_folder+"preprocessed/"

        # shutil.rmtree(args.output_folder, ignore_errors=True)
        # os.makedirs(args.output_folder, exist_ok=True)

        users = os.listdir(signals_directory)
        users = list(filter(lambda x: os.path.isdir(os.path.join(signals_directory, x)), users))

        fun_chain = [
            {'name': 'rolling_average', 'params': {'window_size_seconds': 1.01}}, 
            {'name': 'subtract', 'params': {}}, 
            {'name': 'butter_lowpass_filter', 'params': {'filter_order': 2, 'low': 4}}, 
            {'name': 'cut_start', 'params': {'seconds': 3}}]

        # print(users)

        for i, user in enumerate(sorted(users)):

            print(i, user)

            # get all files
            user_fold = os.path.join(signals_directory, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            # only retain allowed formats
            user_files = filter(lambda x: x.split(".")[-1] == "csv", user_files)
            os.makedirs(os.path.join(output_folder, user), exist_ok=True)

            for file in sorted(user_files):
                filepath = os.path.join(user_fold, file)
                fname = file.split(".")[0]

                csv_fpath = os.path.join(output_folder, user, fname + ".csv")
                
                extracted = pd.read_csv(filepath, index_col=False)
                
                preprocessed = [] 
                columns = []

                sys.stdout.write("{} ({}/{}),{}\n".format(user, i + 1, len(users), file))
                sys.stdout.flush()

                # name_at_step_j = [base_signal_name]
                # print('okok')

                ### applying the filter pipeline here
                signal_data, col_names = [], []

                base_signal_type = 'luma_mean'
                signal_at_step_j = [extracted[base_signal_type].values]
                name_at_step_j = [base_signal_type]

                for j, fun_dict in enumerate(fun_chain):
                    fun = getattr(self, fun_dict["name"])
                    filtered_j = fun(
                        signal_at_step_j[-1],
                        prev_x=signal_at_step_j[-2] if len(signal_at_step_j)>1 else None,
                        **fun_dict["params"]
                    )
                    new_name = "%s>%s" % (name_at_step_j[-1], fun_dict["name"])

                    if len(filtered_j) == len(extracted[base_signal_type].values):
                        signal_at_step_j.append(np.real(filtered_j))
                        name_at_step_j.append(new_name)
                    else:
                        signal_data.append(filtered_j)
                        col_names.append(new_name)
                
                preprocessed.extend(signal_at_step_j)
                columns.extend(name_at_step_j)

                preprocessed = np.array(preprocessed)
                df = pd.DataFrame(preprocessed.T, columns=columns)

                print(preprocessed.shape)
                
                df.to_csv(csv_fpath, sep=",", float_format="%.8f", index=False)

class BeatProcessor():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def mov_avg_flat(self, x, window_size=10):
        conv = np.convolve(x, np.ones(window_size), 'valid') / window_size
        return np.pad(conv, [((window_size - 1) // 2, (window_size - 1) // 2)], mode='constant', constant_values=0)

    def match_mins(self, target_signal, target_signal_idxs, filtered_signal_idxs, min_f_gap):
        t_indexes = []
        for minimum in filtered_signal_idxs:
            start_i = max(0, minimum - min_f_gap)
            actual_offset = minimum - min_f_gap if minimum - min_f_gap >= 0 else 0
            end_i = min(target_signal.shape[0], minimum + min_f_gap)
            subset = target_signal[start_i: end_i]
            argmin = np.argwhere(subset == subset.min())
            argmin = argmin[0, 0]
            idx_in_tar_signal = argmin + actual_offset
            if idx_in_tar_signal in target_signal_idxs:
                t_indexes.append(argmin + actual_offset)
        return np.array(t_indexes)

    def remove_consecutive_mins(self, indexes, signal, min_frame_gap):
        findexes = []
        i = 0
        while i < len(indexes) - 1:
            curr_index = indexes[i]
            next_index = indexes[i + 1]
            if np.abs(curr_index - next_index) >= min_frame_gap:
                findexes.append(curr_index)
                i += 1
            else:
                streak = [curr_index]
                while np.abs(next_index - curr_index) < min_frame_gap and i < len(indexes) - 1:
                    streak.append(next_index)
                    i += 1
                    next_index = indexes[i]
                streak = np.array(streak)
                values_at_indx = signal[streak]
                minim_value_indx = np.argwhere(values_at_indx == values_at_indx.min())[0, 0]
                findexes.append(streak[minim_value_indx])
        findexes.append(indexes[-1])
        return findexes

    # max_bpm, order, processed_column_name, smooth_window_size

    def hb_argrelmin(self, df, order, max_bpm, processed_column_name, smooth_window_size):
        # max_bpm = max_bpm
        frame_rate = self.sample_rate

        min_frame_gap = frame_rate / (max_bpm / 60)

        target_signal = df[processed_column_name].dropna().values
        smoothed_signal = self.mov_avg_flat(target_signal, window_size=smooth_window_size)
        indexes = argrelmin(smoothed_signal, order=order)[0]


        # remove duplicates
        indexes = np.unique(indexes)

        findexes = self.remove_consecutive_mins(indexes, smoothed_signal, min_frame_gap)
        
        # remove duplicates
        findexes = np.unique(findexes)

        mins_in_target_signal = argrelmin(target_signal, order=order)[0]
        findexes = self.match_mins(target_signal, mins_in_target_signal, findexes, int(min_frame_gap / 2))

        # remove duplicates
        findexes = np.unique(findexes)

        all_beats = []
        
        for i, start_beat in enumerate(findexes[:-1]):
            end_beat = findexes[i + 1] + 1
            subsig = target_signal[start_beat:end_beat]
            all_beats.append(subsig)
        
        return all_beats

    def separate_beats(self, preprocessed_directory, output_folder):
        users = os.listdir(preprocessed_directory)
        users = list(filter(lambda x: os.path.isdir(os.path.join(preprocessed_directory, x)), users))

        for i, user in enumerate(sorted(users)):
            ## get all files
            user_fold = os.path.join(preprocessed_directory, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            user_files = filter(lambda x: x.split(".")[-1] == "csv", user_files)
            os.makedirs(os.path.join(output_folder, user), exist_ok=True)

            for file in sorted(user_files):
                filepath = os.path.join(user_fold, file)
                fname = file.split(".")[0]
                
                json_fpath = os.path.join(output_folder, user, fname + ".json")
                
                extracted_s = pd.read_csv(filepath, index_col=False)
                
                print(user, file)
                
                beats = dict()
                
                good_beats = self.hb_argrelmin(
                    df=extracted_s,
                    max_bpm=101,
                    order=2,
                    processed_column_name='luma_mean>rolling_average>subtract>butter_lowpass_filter>cut_start',
                    smooth_window_size=13
                )

                extraction_type = 'hb_argrelmin'
                beats[extraction_type] = dict()
                for j, b in enumerate(good_beats):
                    beats[extraction_type][j] = b.tolist()

                ## save data to json for later use
                json.dump(beats, open(json_fpath, "w"))

    def mov_avg(self, x, n):
        return np.convolve(x, np.ones(n), 'same') / n

    def first_minmax_if_exists(self, x, kind):
        if kind == "min":
            mins = argrelmin(x)[0]
            if len(mins)>0:
                return mins[0]
            else:
                return 0
        if kind == "max":
            maxs = argrelmax(x)[0]
            if len(maxs) > 0:
                return maxs[0]
            else:
                return 0

    def hb_fp_detection(self, beats, **kwargs):

        results = {}

        for i, b in enumerate(beats):

            _b = np.array(b)
            if len(_b) < 25:
                _b = np.pad(_b, (0, 25-len(_b)), mode="edge")
            _1b = self.mov_avg(utils.derivative(np.array(b), 1), 25)
            _2b = utils.derivative(np.array(_b), 2)

            assert len(_1b) == len(_2b) == len(_b), "%d, %d, %d" % (len(_b), len(_1b), len(_2b))

            _b -= _b.min()

            _b /= _b.max()

            # first point of interest is maximum in 1st dev
            p1 = np.argmax(_1b)
            # if there is a minimum that is not in index 0, cut it
            _1b_before_p1 = _1b[:p1]
            mins_before_p1 = argrelmin(_1b_before_p1)[0]
            if len(mins_before_p1) > 0:
                _b = _b[mins_before_p1[-1]:]
                _1b = _1b[mins_before_p1[-1]:]
                _2b = _2b[mins_before_p1[-1]:]
                p1 = p1 - mins_before_p1[-1]

            # define sys peak as first maximum after p1 in _b
            p2 = argrelmax(_b[p1:])[0][0] + p1
            # or the first minimum in _1b , whichever comes first
            p2 = min(p2, self.first_minmax_if_exists(_1b[p1:], "min")+p1)

            # first min after p2 in _1b
            p3 = self.first_minmax_if_exists(_1b[p2:], "min") + p2
            # first max after p3 in _1b
            p4 = p3
            p5 = p4
            maxes_after_p3 = argrelmax(_1b[p3:])[0]
            if len(maxes_after_p3)>0:
                p4 = maxes_after_p3[0] + p3
                mins_after_p4 = argrelmin(_1b[p4:])[0]
                if len(mins_after_p4) > 0:
                    p5 = mins_after_p4[0] + p4

            # adding sanity checks
            # defaulting to moot indexes if the function fails to locate fiducial points
            p1 = max(0, min(int(p1), len(_b) - 1))
            p2 = max(0, min(int(p2), len(_b) - 1))
            p3 = max(0, min(int(p3), len(_b) - 1))
            p4 = max(0, min(int(p4), len(_b) - 1))
            p5 = max(0, min(int(p1), len(_b) - 1))

            results[i] = {}
            results[i]["_b"] = _b.tolist()
            results[i]["_1b"] = _1b.tolist()
            results[i]["_2b"] = _2b.tolist()
            results[i]["p1"] = p1
            results[i]["systolic_peak_i"] = p2
            results[i]["systolic_peak_c"] = 1.0
            results[i]["dychrotic_notch_i"] = p3
            results[i]["dychrotic_notch_c"] = 1.0
            results[i]["diastolic_peak_i"] = p4
            results[i]["diastolic_peak_c"] = 1.0
            results[i]["p5"] = p5

        return results

    def detect_FDs(self, beats_directory, output_folder):
        # pbeats_directory
        # pass
    
        users = os.listdir(beats_directory)
        users = list(filter(lambda x: os.path.isdir(os.path.join(beats_directory, x)), users))

        for i, user in enumerate(sorted(users)):
            # get all files
            user_fold = os.path.join(beats_directory, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            # only retain allowed formats
            user_files = filter(lambda x: x.split(".")[-1] == "json", user_files)
            os.makedirs(os.path.join(output_folder, user), exist_ok=True)
            for file in sorted(user_files):
                filepath = os.path.join(user_fold, file)
                fname = file.split(".")[0]

                json_out_fpath = os.path.join(output_folder, user, fname + ".json")
                # if not os.path.isfile(json_out_fpath) or not os.path.isfile(img_out_fpath):
                
                extracted_beats = json.load(open(filepath, "r"))
                
                print(user, file)
                
                keys = sorted(map(int, extracted_beats["hb_argrelmin"].keys()))
                list_of_beats = [extracted_beats["hb_argrelmin"][str(k)] for k in keys]

                r = self.hb_fp_detection(list_of_beats)
                json.dump(r, open(json_out_fpath, "w"))

class FeatureProcessor():

    def __init__(self, ):
        pass

    def derivative(self, signal, index=1):
        # Implemented according to [1]
        if index == 1:
            return [a - b for a, b in zip(signal, signal[1:])] + [0]
        elif index == 2:
            return [0] + [c + a - 2 * b for a, b, c in zip(signal, signal[1:], signal[2:])] + [0]
        else:
            raise ValueError("Only support first or second derivatives")
            
    def f_fiducial_points(self, beats, beat_det_names, fiducial_points, interp_dim):

        feature_names = [
            "meta_counter",
            "peak_to_peak_t",  # From [1] as delta T
            "systolic_peak_index",  # From [1], as y
            "dychrotic_notch_index",   # From [1] as t1, time to first peak
            "diastolic_peak_index",   # From [1] as t3, diastolic peak index
            "A2_area",  # From [1], but instead of notch we do to diastolic peak
            "A1_area",  # From [1], but we do from diastolic peak down as opposed to notch down
            "A2_A1_ratio",
            "a1",   # Maximum of first derivative
            "b1",   # Minimum of first derivative
            "ta1",  # Index of a1
            "tb1",  # index of b1
            "a2",   # Maximum value of second derivative
            "b2",   # Minimum value of second derivative
            "ta2",  # Index of a2
            "tb2",  # Index of b2
            "b2_a2",     # b2 / a2
            "systolic_peak_c",
            "dychrotic_notch_c",
            "diastolic_peak_c"
        ]

        features = []

        for beat_det_name in beat_det_names:

            for beat_counter in beats[beat_det_name].keys():
                this_b_features = [beat_counter]
                signal = np.array(beats[beat_det_name][beat_counter])
                signal -= signal.min()
                signal /= signal.max()
                x, y = interpolate_beat(signal, interp_dim)

                scale_factor = interp_dim/signal.shape[0]

                systolic_peak_index = int(fiducial_points[beat_counter]["systolic_peak_i"]*scale_factor)
                systolic_peak_conf = fiducial_points[beat_counter]["systolic_peak_c"]
                # systolic_peak_value = y[systolic_peak_index]

                dychrotic_notch_index = int(fiducial_points[beat_counter]["dychrotic_notch_i"]*scale_factor)
                dychrotic_notch_conf = fiducial_points[beat_counter]["dychrotic_notch_c"]
                # dychrotic_notch_value = y[dychrotic_notch_index]

                diastolic_peak_index = int(fiducial_points[beat_counter]["diastolic_peak_i"]*scale_factor)
                diastolic_peak_conf = fiducial_points[beat_counter]["diastolic_peak_c"]
                # diastolic_peak_value = y[diastolic_peak_index]

                peak_to_peak = np.abs(diastolic_peak_index - systolic_peak_index)

                a1_area = np.trapz(y[:dychrotic_notch_index])
                a2_area = np.trapz(y[dychrotic_notch_index:])
                area_ratio = a2_area/a1_area

                first_deriv = self.derivative(y, 1)
                second_deriv = self.derivative(y, 2)
                a1 = np.max(first_deriv)
                b1 = np.min(first_deriv)
                ta1 = np.argmax(first_deriv)
                tb1 = np.argmin(first_deriv)
                a2 = np.max(second_deriv)
                b2 = np.min(second_deriv)
                ta2 = np.argmax(second_deriv)
                tb2 = np.argmin(second_deriv)
                b2_a2 = b2 / a2

                this_b_features.extend([
                    peak_to_peak,
                    systolic_peak_index,
                    dychrotic_notch_index,
                    diastolic_peak_index,
                    a1_area,
                    a2_area,
                    area_ratio,
                    a1,
                    b1,
                    ta1,
                    tb1,
                    a2,
                    b2,
                    ta2,
                    tb2,
                    b2_a2
                ])

                this_b_features.extend([
                    systolic_peak_conf,
                    dychrotic_notch_conf,
                    diastolic_peak_conf
                ])

                features.append(this_b_features)

        features = np.array(features)
        if features.shape[0]==0:
            features = np.empty((0, len(feature_names)))
        assert features.shape[1] == len(feature_names), "%d, %d" % (features.shape[1], len(feature_names))
        return features, feature_names

    def extract_feats(self, output_folder, peaks_folder, fiducial_points_folder):
        os.makedirs(output_folder, exist_ok=True)
        utils.delete_all_subdirs(output_folder)

        users = os.listdir(peaks_folder)
        users = list(filter(lambda x: os.path.isdir(os.path.join(peaks_folder, x)), users))

        for i, user in enumerate(sorted(users)):
            # get all files
            user_fold = os.path.join(peaks_folder, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            # only retain allowed formats
            user_files = filter(lambda x: x.split(".")[-1] == "json", user_files)

            for file in sorted(user_files):
                filepath = os.path.join(user_fold, file)
                fiducial_points_file = os.path.join(fiducial_points_folder, user, file)

                fname = file.split(".")[0]
                
                ext_folder_name = 'fiducial_points'

                os.makedirs(os.path.join(output_folder, ext_folder_name, user), exist_ok=True)
                csv_out_fpath = os.path.join(output_folder, ext_folder_name, user, fname + ".csv")
                extracted_beats = json.load(open(filepath, "r"))
                fiducial_points = json.load(open(fiducial_points_file, "r"))
                
                interp_dim = 200
                
                features, fnames = self.f_fiducial_points(extracted_beats, ["hb_argrelmin"], fiducial_points=fiducial_points, interp_dim=interp_dim)
                df = pd.DataFrame(features, columns=fnames)
                df.to_csv(csv_out_fpath, sep=",", float_format="%.6f", index=False)
                

        # now merge into single file
        utils.merge_features_into_file(output_folder, postfix='')

        # delete all temp subdirectories in /home/data/features
        utils.delete_all_subdirs(output_folder)

if __name__ == "__main__":

    root_folder = 'C:/Users/numan98khan/Desktop/HemaRays/research/HemaRays_Data_OneFourth/'
    out_ext = 'dump/'

    out_fold = root_folder + out_ext

    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    e = VideoProcessor(sample_rate=30, 
        super_folder=out_fold)
    
    bd = BeatProcessor(sample_rate=30)
    
    fe = FeatureProcessor()
    

    # e.extract_signal(save=False, vid_dir=root_folder)
    # e.preprocess_signal()

    # bd.separate_beats(preprocessed_directory = out_fold+"preprocessed/",
    #     output_folder = out_fold+"beat_sepped/"
    # )

    # bd.detect_FDs(beats_directory = out_fold+"beat_sepped/",
    #     output_folder = out_fold+"fd_points/"
    # )

    # fe.extract_feats(output_folder = out_fold+"features/",
    #     peaks_folder = out_fold+"beat_sepped/",
    #     fiducial_points_folder = out_fold+"fd_points/"
    # )



