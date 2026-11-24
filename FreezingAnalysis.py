# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

from pynwb import NWBHDF5IO
from turner_suite import plotting
from turner_suite.nwb import sleap, nwb_analysis

from scipy.ndimage import sum_labels, label
from scipy.stats import mannwhitneyu
plt.style.use('seaborn-v0_8-poster')

plotting.set_lab_plotstyle()

# data_dir = '/mnt/chromeos/SMB/beb0cbe83aa7a1d01f81b848eb4b574884a81f51762ed5a4731183ab8eb7d321/ad/PLExperiments/2026-07-09'
# data_dir = '/network/rit/lab/turner_lab/ad/KirExperiments/2026-07-23'
# data_dir = '/home/turnerlab/data/ad/KirExperiments/2026-07-23'
# data_dir = '/network/rit/lab/turner_lab/ad/PLExperiments/2026-07-09'

data_dir = '/home/mhturner/Data/FreezingExperiments/BehaviorData'

matching_filepaths = nwb_analysis.filter_datafiles(
    data_dir,
    subject_dict={'strain': 'Canton-S'},
    parameter_dict={
        'protocol_id': 'FlatCircularMovingSpot',
        'spot_radius': 0.0015
    },
    recursive_search=False
)

matching_filepaths = sorted(list(set(matching_filepaths)))
print(f"Found {len(matching_filepaths)} Canton-S NWB files with 0.0015 stimulus.")

def fill_missing_tracks(pos_array):
     """
     Fills in missing tracks from SLEAP putout

     returns filled in tracks
     """
     df = pd.DataFrame(pos_array)
     df_filled = df.interpolate(method="linear", limit=5)
     df_filled = df_filled.ffill().bfill()
     return df_filled.to_numpy()

fps = 106.76  # Recording frame rate
min_freeze_frames = int(0.15 * fps)

population_stats = []
raster_data = []

for filepath in matching_filepaths:
     # Retrieve filename and IDs for both flies
     nwb_filename = os.path.basename(filepath)
     print(f"Processing: {nwb_filename}")

     fly_id = nwb_filename.split('_')[1]
     
     with NWBHDF5IO(filepath, "r") as nwb_io:
          nwb_file = nwb_io.read()
          sm = sleap.get_sleap_module(nwb_file)

          # Retrieve head position for male
          timestamps, pos_male_head = sleap.get_position(nwb_file, node='head', track_name=fly_id)
          pos_male_head = fill_missing_tracks(pos_male_head)

          # Convert m -> cm
          pos_male_head *= 100

          # Calculate speed (cm/s)
          dt = np.diff(timestamps)
          male_step = np.linalg.norm(np.diff(pos_male_head, axis=0), axis=1)

          male_speed = male_step / dt

          male_speed[male_speed > 6.0] = np.nan

          window_size = 5
          smoothed_male = pd.Series(male_speed).rolling(window=window_size, min_periods=1).mean()

          n_speed_frames = len(male_speed)
          time_axis_speed = (timestamps - timestamps[0])[:n_speed_frames]
          speed_vals = smoothed_male.values[:n_speed_frames] if hasattr(smoothed_male, 'values') else smoothed_male[:n_speed_frames]

          # Calculate freezing behavior and see if the behavior is robust
          # Use Speed of Drosophila plots & trial zones
               # View speed during pre time and during stim time
               # Trialize time series
               # Avg velocity across all trials
                    # Use trialize function
                    # Take avg of instances/trials where freezing occurs
                    # Avg of trials where freezing does not occur
                    # Check if freezing is being properly classified
                         # Zoom in on certain trials to see better
                    # Chop up trial responses (look individually, look at means)
               # A fly was considered to be freezing when average motion around the fly was lower than 50 pixels s−1 (~5% of fly area)
               # Include stimulus tracking in data (sleap_vis_stim.py)
          # Protocol timing: 4s pre-time -> 1s stim -> 1s tail time

          trials_df = nwb_file.intervals['trials'].to_dataframe()
          trial_timestamps, m_trialized = nwb_analysis.trialize_timeseries(pos_male_head, timestamps, trials_df, query=None, baseline_subtract=False)
          n_trials = len(trials_df)

          pre_speed_per_trial = np.full(n_trials, np.nan)
          stim_speed_per_trial = np.full(n_trials, np.nan)
          
          pre_speeds_list = []
          stim_speeds_list = []
          
          trial_frozen_points = np.zeros(len(trials_df), dtype=bool)
          # mean_stim_speeds_all = []

          for t, row in trials_df.iterrows():
               onset_time = row['start_time'] # Stimulus onset timestamp

               # Define time boundaries
               pre_start = onset_time - 4.0
               pre_end = onset_time
               stim_start = onset_time
               stim_end = onset_time + 1.0

               pre_mask = (time_axis_speed >= pre_start) & (time_axis_speed < pre_end)
               stim_mask = (time_axis_speed >= stim_start) & (time_axis_speed < stim_end)

               trial_pre_speed = np.nan
               if np.any(pre_mask) and not np.all(np.isnan(speed_vals[pre_mask])):
                    trial_pre_speed = np.nanmean(speed_vals[pre_mask])
                    pre_speeds_list.append(trial_pre_speed)
               pre_speed_per_trial[t] = trial_pre_speed

               # The fly MUST be actively moving before stim (>= 0.5 cm/s) to qualify for assessing a stimulus-evoked freezing response.
               is_fly_active_before_stim = (not np.isnan(trial_pre_speed)) and (trial_pre_speed >= 0.5)

               t_mean_stim_speed = np.nan # reset each trial so a stale value isnt reused
               if np.any(stim_mask) and not np.all(np.isnan(speed_vals[stim_mask])):
                    t_stim_speeds = speed_vals[stim_mask]
                    t_mean_stim_speed = np.nanmean(t_stim_speeds)
                    stim_speeds_list.append(t_mean_stim_speed)

                    if is_fly_active_before_stim: 
                         is_lowspeed = (t_stim_speeds < 0.5) & (~np.isnan(t_stim_speeds))
                         labeled_instances, num_instances = label(is_lowspeed)

                         if num_instances > 0:
                              instance_sizes = sum_labels(is_lowspeed, labeled_instances, index=np.arange(1, num_instances + 1))
                              if np.any(instance_sizes >= min_freeze_frames): # If statment to detect if any of the moments of low speed matches freezing criteria
                                   trial_frozen_points[t] = True
                                   # Record relative time points where freezing occurs within eah trial for raster plot
                                   sub_timestamps = time_axis_speed[stim_mask]
                                   valid_instances = np.where(instance_sizes >= min_freeze_frames)[0] + 1
                                   for instance in valid_instances:
                                        freeze_frame_mask = (labeled_instances == instance)
                                        freeze_times = sub_timestamps[freeze_frame_mask]
                                        raster_data.append({
                                             'filename': nwb_filename,
                                             'fly_id': fly_id,
                                             'trial_num': t + 1,
                                             'freeze_times': freeze_times
                                        })
               stim_speed_per_trial[t] = t_mean_stim_speed
               
          avg_pre_speed = np.nanmean(pre_speeds_list) if pre_speeds_list else np.nan
          avg_stim_speed = np.nanmean(stim_speeds_list) if stim_speeds_list else np.nan
          speed_drop_ratio = (avg_stim_speed / avg_pre_speed) if (avg_pre_speed > 0 and not np.isnan(avg_pre_speed)) else 1.0
          pct_trials_frozen = np.mean(trial_frozen_points) * 100   

          # Habitution split: Compare Q1 (Trials 1-25) vs Q4 (Trials 76-100)
          q1_trials = trial_frozen_points[:25]
          q4_trials = trial_frozen_points[75:100]
          q1_pct = np.mean(q1_trials) * 100 if len(q1_trials) > 0 else np.nan
          q4_pct = np.mean(q4_trials) * 100 if len(q4_trials) > 0 else np.nan

          stim_speed_array = np.array(stim_speed_per_trial)
         
          # Avg velocity of flies in trials where freezing occurs
          avg_speed_freezing_trials = np.nanmean(stim_speed_array[trial_frozen_points]) if np.any(trial_frozen_points) else np.nan
          
          # Avg velocity of flies in trials where freezing doesn't occurs
          avg_speed_nonfreezing_trials = np.nanmean(stim_speed_array[~trial_frozen_points]) if np.any(~trial_frozen_points) else np.nan
         
          # Select only the pre stimulus zones that come before a trial containing freezing
          avg_prestim_freezing_trials = (
               np.nanmean(pre_speed_per_trial[trial_frozen_points])
               if np.any(trial_frozen_points) else np.nan
          )

          # Trialize velocity and plot trace to see velocity changes when stimulus is on/off
          vtrial_timestamps, v_trialized = nwb_analysis.trialize_timeseries(speed_vals, time_axis_speed, trials_df, query=None, baseline_subtract=False)
          rel_time = (vtrial_timestamps - vtrial_timestamps[0]) - 4.0

          n_valid_trials = min(v_trialized.shape[0], m_trialized.shape[0])
          v_valid = v_trialized[:n_valid_trials]

          # Change in speed relative to each trial's pre-stimulus baseline (-4s to 0s)
          pre_win_mask = (rel_time >= -4.0) & (rel_time < 0.0)
          trial_baseline_speed = np.nanmean(v_valid[:, pre_win_mask], axis=1, keepdims=True)
          delta_v_trialized = v_valid - trial_baseline_speed


          population_stats.append({
               'filename': nwb_filename,
               'fly_id': fly_id,
               'strain': 'Canton-S',
               'avg_pre_speed (cm/s)': avg_pre_speed,
               'avg_stim_speed (cm/s)': avg_stim_speed,
               'speed_drop_ratio': speed_drop_ratio,
               'pct_trials_frozen (%)': pct_trials_frozen,
               'Q1_frozen_pct': q1_pct,
               'Q4_frozen_pct': q4_pct,
               'avg_speed_frozen_trials (cm/s)': avg_speed_freezing_trials,
               'avg_speed_nonfrozen_trials (cm/s)': avg_speed_nonfreezing_trials,
               'avg_prestim_freezing_trials (cm/s)': avg_prestim_freezing_trials
          })

          # Speed of Drosophila Plot  
          fh2, ax2 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
          ax2.plot(time_axis_speed, smoothed_male, label="Male Speed", color='#1976D2', alpha=0.7)
          ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
          ax2.set_ylabel("Speed (cm/s)", fontsize=12, fontweight='bold')

          # (Raster Plot) Plot ticks that show where freezing occurs in the recordings at the bottom of the speed plot
          file_freezing_timestamps = []
          fly_rasters = [r for r in raster_data if r['filename'] == nwb_filename]
          for hit in fly_rasters:
               file_freezing_timestamps.extend(hit['freeze_times'])

          if len(file_freezing_timestamps) > 0:
               file_freezing_timestamps = np.array(file_freezing_timestamps)
               tick_y_freeze = -0.3
               ax2.scatter(
                    file_freezing_timestamps,
                    np.full(len(file_freezing_timestamps), tick_y_freeze),
                    color="purple", marker="|", s=150, linewidths=2.0, label="Freezing"
               )
               ax2.set_ylim(-0.6, ax2.get_ylim()[1])

          # Align trial windows with actual recording to increase visual accuracy in plots
          for idx, row in trials_df.iterrows():
               onset = row['start_time']
               offset = onset + 1.0  # 1s stimulus duration
               if offset <= time_axis_speed[-1]:
                    lbl = 'Stimulus Active (1s)' if idx == 0 else ""
                    ax2.axvspan(onset, offset, color='gold', alpha=0.15, label=lbl)

          ax2.set_xlim(0, time_axis_speed[-1]) # Hard set x-axis

          ax2.legend(loc='upper right', frameon=True)
          plt.tight_layout()
          
          # Visualize Stim Plots by Trials
          # fh7, ax7 = plt.subplots(4, 3, figsize=(6, 8))
          #           ax7 = ax7.ravel()
          #           [x.set_xlim([1, 7]) for x in ax7]
          #           [x.set_ylim([-1, 5]) for x in ax7]
          #           [x.set_aspect('equal') for x in ax7]
                    
          #           for t in range(m_trialized.shape[0]):
          #                spot_xy = sleap.get_spot_position(nwb_file=nwb_file, trial_index=t, timestamps=trial_timestamps, track_name=fly_id)
          #                spot_xy *= 100  # meters -> cm
                    
          #                ax7[t].plot(m_trialized[t, :, 0], m_trialized[t, :, 1], 'b-')
          #                ax7[t].plot(spot_xy[:, 0], spot_xy[:,  1], 'k-')
          #                ax7[t].set_title(f'Trial {t}')

          freezing_trial_indices = np.where(trial_frozen_points)[0]
          
          if len(freezing_trial_indices) > 0:
               # Pick up to 12 exemplar freezing trials to plot
               exemplar_trials = freezing_trial_indices[:12]
               fh7, ax7 = plt.subplots(4, 3, figsize=(6, 8))
               ax7 = ax7.ravel()
               [x.set_xlim([1, 7]) for x in ax7]
               [x.set_ylim([-1, 5]) for x in ax7]
               [x.set_aspect('equal') for x in ax7]

               for i, t_idx in enumerate(exemplar_trials):
                         if i >= len(ax7):
                              break
                         spot_xy = sleap.get_spot_position(nwb_file=nwb_file, trial_index=t_idx, timestamps=trial_timestamps, track_name=fly_id)
                         spot_xy *= 100  # meters -> cm
                         
                         ax7[i].plot(m_trialized[t_idx, :, 0], m_trialized[t_idx, :, 1], 'b-')
                         ax7[i].plot(spot_xy[:, 0], spot_xy[:,  1], 'k-')
                         ax7[i].set_title(f'Trial {t_idx}')

               # Turn off any unused subplots if fewer than 12 freezing trials exist
               for j in range(len(exemplar_trials), len(ax7)):
                    ax7[j].axis('off')

               plt.tight_layout()

          # Speed Trace Trialized
          mean_v = np.nanmean(v_valid, axis=0)
          sem_v = np.nanstd(v_valid, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(v_valid), axis=0))

          fh_tr, ax_tr = plt.subplots(figsize=(8, 4), dpi=300)
          ax_tr.plot(rel_time, mean_v, color='#1976D2', linewidth=2, label='Mean Speed')
          ax_tr.fill_between(rel_time, mean_v - sem_v, mean_v + sem_v, color='#1976D2', alpha=0.25, label='± SEM')
          ax_tr.axvspan(0.0, 1.0, color='gold', alpha=0.2, label='Stimulus (1s)')
          ax_tr.axvline(0.0, color='k', linestyle='--', linewidth=1.0)
          ax_tr.set_xlabel("Time Relative to Stimulus Onset (s)", fontsize=12, fontweight='bold')
          ax_tr.set_ylabel("Speed (cm/s)", fontsize=12, fontweight='bold')
          ax_tr.set_title(f"Trial-Averaged Speed — {fly_id}", fontsize=12, fontweight='bold')
          ax_tr.set_ylim(bottom=0)
          ax_tr.legend(loc='upper right', frameon=True)
          plt.tight_layout()

df_freezing_pop = pd.DataFrame(population_stats)
print("\n--- Population Freezing Summary ---")
print(df_freezing_pop)

# Save summary to CSV
save_dir = os.path.join(data_dir, "freezing_plots")
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "CantonS_Freezing_Population_Summary.csv")
df_freezing_pop.to_csv(csv_path, index=False)
print(f"Saved summary CSV to {csv_path}")

# %%

# Paired plot showing difference between speed in trials where freezing occurs vs speed in trials where freezing doesn't occur
fh, ax = plt.subplots(figsize=(7, 4))

df_melted = pd.melt(
    df_freezing_pop, 
    id_vars=['fly_id'], 
    value_vars=['avg_speed_frozen_trials (cm/s)', 'avg_speed_nonfrozen_trials (cm/s)'],
    var_name='Trial Condition', 
    value_name='Speed of Drosophila (cm/s)'
)
df_melted['Trial Condition'] = df_melted['Trial Condition'].replace({'avg_speed_frozen_trials (cm/s)': 'Freezing', 'avg_speed_nonfrozen_trials (cm/s)': 'No Freezing'})

# Violin Plot for population distribution
sns.violinplot(
     data=df_melted,
     x="Trial Condition",
     y="Speed of Drosophila (cm/s)",
     hue="Trial Condition",
     split=False,
     cut=0,
     inner="quartile",
     palette="muted",
     legend=False,
     ax=ax
)
    
# Individual data points scattered on top
sns.stripplot(
     data=df_melted,
     x="Trial Condition",
     y="Speed of Drosophila (cm/s)",
     hue="Trial Condition",
     color="black",
     alpha=0.7,
     size=6,
     jitter=0.15,
     ax=ax
)

# paired plot showing the potential shift in freezing percentage between the first and last quarters of the session to test for habituation.
fh, ax = plt.subplots(figsize=(7, 4))

df_melted = pd.melt(
    df_freezing_pop, 
    id_vars=['fly_id'], 
    value_vars=['Q1_frozen_pct', 'Q4_frozen_pct'],
    var_name='Quarter', 
    value_name='Freezing_Pct'
)
df_melted['Quarter'] = df_melted['Quarter'].replace({'Q1_frozen_pct': 'Quarter 1 (1-25)', 'Q4_frozen_pct': 'Quarter 4 (76-100)'})

# Violin Plot for population distribution
sns.violinplot(
     data=df_melted,
     x="Quarter",
     y="Freezing_Pct",
     hue="Quarter",
     split=False,
     cut=0,
     inner="quartile",
     palette="muted",
     legend=False,
     ax=ax
)
    
# Individual data points scattered on top
sns.stripplot(
     data=df_melted,
     x="Quarter",
     y="Freezing_Pct",
     hue="Quarter",
     color="black",
     alpha=0.7,
     size=6,
     jitter=0.15,
     ax=ax
)

population_stats.append({
               'filename': nwb_filename,
               'fly_id': fly_id,
               'strain': 'Canton-S',
               'avg_pre_speed (cm/s)': avg_pre_speed,
               'avg_stim_speed (cm/s)': avg_stim_speed,
               'speed_drop_ratio': speed_drop_ratio,
               'pct_trials_frozen (%)': pct_trials_frozen,
               'Q1_frozen_pct': q1_pct,
               'Q4_frozen_pct': q4_pct,
               'avg_speed_frozen_trials (cm/s)': avg_speed_freezing_trials,
               'avg_speed_nonfrozen_trials (cm/s)': avg_speed_nonfreezing_trials,
               'avg_prestim_freezing_trials (cm/s)': avg_prestim_freezing_trials
          })

# paired plot showing shift in speed between prestim and stim when freezing occurs
fh, ax = plt.subplots(figsize=(7, 4))

df_melted = pd.melt(
    df_freezing_pop, 
    id_vars=['fly_id'], 
    value_vars=['avg_prestim_freezing_trials (cm/s)', 'avg_speed_frozen_trials (cm/s)'],
    var_name='Stimulus', 
    value_name='Speed of Drosophila (cm/s)'
)
df_melted['Stimulus'] = df_melted['Stimulus'].replace({'avg_prestim_freezing_trials (cm/s)': 'Pre', 'avg_speed_frozen_trials (cm/s)': 'Stim'})

# Violin Plot for population distribution
sns.violinplot(
     data=df_melted,
     x="Stimulus",
     y="Speed of Drosophila (cm/s)",
     hue="Stimulus",
     split=False,
     cut=0,
     inner="quartile",
     palette="muted",
     legend=False,
     ax=ax
)
    
# Individual data points scattered on top
sns.stripplot(
     data=df_melted,
     x="Stimulus",
     y="Speed of Drosophila (cm/s)",
     hue="Stimulus",
     color="black",
     alpha=0.7,
     size=6,
     jitter=0.15,
     ax=ax
)


# %% Mann-Whitney U Test
g1_data = df_freezing_pop['avg_speed_frozen_trials (cm/s)']
g2_data = df_freezing_pop['avg_speed_nonfrozen_trials (cm/s)']

stat, p_val = mannwhitneyu(g1_data, g2_data, alternative='two-sided')

print(f"{stat} and {p_val}")

g3_data = df_freezing_pop['Q1_frozen_pct']
g4_data = df_freezing_pop['Q4_frozen_pct']

stat1, p_val1 = mannwhitneyu(g3_data, g4_data, alternative='two-sided')

print(f"{stat1} and {p_val1}")

g5_data = df_freezing_pop['avg_prestim_freezing_trials (cm/s)']
g6_data = df_freezing_pop['avg_speed_frozen_trials (cm/s)']

stat2, p_val2 = mannwhitneyu(g5_data, g6_data, alternative='two-sided')

print(f"{stat2} and {p_val2}")

# Are they slowing down during the stimulus?
## See if there is a decrease in speed (1 second before stimulus onset and during the stimulus)
# %%
