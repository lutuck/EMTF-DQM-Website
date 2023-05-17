#imports here
import os
import numpy as np
import pandas as pd
from requests_futures.sessions import FuturesSession
import uproot
import ROOT
import runregistry
from multiprocessing.pool import ThreadPool
import requests
from requests_file import FileAdapter   # allows local files w requests

def main(run_nums, plot_types, rc, ls):
    """
    run numbers to be queried, type of plots to show, run class, lumisections
    """
    print('Searching for plot types: ')
    print(plot_types)
    
    TIMEOUT = 5

    def _parse_run_full_name(full_name):
        """
        returns the simplified form of a full DQM name.
        """
        if VERBOSE >= 2:
            print('\ndqm.py _parse_run_full_name(full_name = %s)' % full_name)

        if full_name.split('_')[2].startswith('R000'):          # OfflineData
            name = full_name.split('_')[2][1:]
            return str(int(name)) # why both?? 
        elif full_name.split('_')[3].startswith('R000'):        # OnlineData
            name = full_name.split('_')[3][1:].replace('.root','')
            return str(int(name))
        else:
            raise ValueError('dqm.py _pars_run_full_name({}), failed to parse run number!'.format(full_name))


    def _fetch_dqm_rows(url, timeout=TIMEOUT):
        """
        Return a future? of DQMRows of a DQM page at url
        Access the array of DQMRows at _resolve(self._fetch_dqm_rows(...)).data
        """

        return sess.get(url, timeout=timeout, verify=sess.verify, stream=True)
    
    runs = [run.strip() for run in run_nums.split(',')] # splits runs by commas, removes extra spaces

    new_runs = []
    for run in runs:
        if ':' in run:                  # allows run range to be entered in xxxxxx:xxxxxx format
            bounds = run.split(':')
            for new_run in range(int(bounds[0]),int(bounds[1])+1):
                new_runs.append(str(new_run))
        else:
            new_runs.append(run)        # otherwise just adds runs, split by commas in previous step
    runs = new_runs                     ## runs is now list of run nums as strings

    runs_int = [int(run) for run in runs]

    print('Run class (Collisions or Cosmics/Comissioning): ')
    if rc == 'Collisions':
        print(rc)
        request = runregistry.get_runs(
                filter = {
                    'class':{
                        'or':[
                            'Collisions23',
                            'Collisions22',
                            'Collisions18'
                        ]
                    },
                    'run_number':{
                        'and':[
                            {'>=': min(runs_int)},
                            {'<=': max(runs_int)}
                        ]
                    }
                }
        )
    else:
        print(rc) # might as well
        request = runregistry.get_runs(
                filter = {
                    'class':{
                        'or':[
                            'Cosmics23',
                            'Cosmics22',
                            'Cosmics18',
                            'Commissioning'
                        ]
                    },
                    'run_number':{
                        'and':[
                            {'>=': min(runs_int)},
                            {'<=': max(runs_int)}
                        ]
                    }
                }
        )

    min_ls_duration = int(ls)
    valid_runs = []
    valid_dates = []
    for run in request:
        if int(run['oms_attributes']['ls_duration']) < min_ls_duration:
            continue
        valid_runs += [str(run['oms_attributes']['run_number'])]
        valid_dates += [str(run['oms_attributes']['start_time'])[5:10]]


    new_runs = []
    dates = []

    for run in runs:
        try:                            # will go through all runs, find indices of valid ones, and add those runs to new lists
            i = valid_runs.index(run)
            new_runs.append(valid_runs[i])
            dates.append(valid_dates[i])
        except:
            print('Skipping run:',str(run))

    runs = new_runs
    print('Valid runs: ')
    print(runs)

    file_names = []
    for run in runs:
        file_names.append(f'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OnlineData/original/000{run[:2]}xxxx/000{run[:4]}xx/DQM_V0001_L1T_R000{run}.root')


    sess = FuturesSession()
    sess.verify = os.environ['CACERT']
    sess.cache = os.environ['CACHE']
    sess.cert = (os.environ['PUBLIC_KEY'], os.environ['PRIVATE_KEY'])


#    full_path = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/Run2022/ZeroBias/0003558xx/DQM_V0001_R000355892__ZeroBias__Run2022C-10Dec2022-v1__DQMIO.root'              # still unsure of the use
#    response = _fetch_dqm_rows(full_path).result()          # need to review this chunk
#    with open(f'/root/csctiming/tmp.root', 'wb') as f:
#        for chunk in response.iter_content(chunk_size=8192):
#            f.write(chunk)

    # going to rewrite above section using a local file to avoid all the SSL madness
    s = requests.Session()
    s.mount('file://', FileAdapter())
    response = s.get('file:///root/csctiming/dqm_full_path.root')
    with open(f'/root/csctiming/tmp.root', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    file = uproot.open('/root/csctiming/tmp.root')


    hits_bx0 = file["DQMData/Run 355892/L1T/Run summary/L1TStage2EMTF/Timing/cscLCTTimingBX0;1"].values()
    hits_bxneg1 = file["DQMData/Run 355892/L1T/Run summary/L1TStage2EMTF/Timing/cscLCTTimingBXNeg1;1"].values()
    hits_bxpos1 = file["DQMData/Run 355892/L1T/Run summary/L1TStage2EMTF/Timing/cscLCTTimingBXPos1;1"].values()
    del file
    os.remove('/root/csctiming/tmp.root')
    hits_bx0 = np.zeros(hits_bx0.shape)
    hits_bxneg1 = np.zeros(hits_bxneg1.shape)
    hits_bxpos1 = np.zeros(hits_bxpos1.shape)
    plots = []
    final_runs = []

    def process_file(idx, fn):
        nonlocal final_runs
        sess = FuturesSession()
        sess.verify = os.environ['CACERT']
        sess.cache = os.environ['CACHE']
        sess.cert = (os.environ['PUBLIC_KEY'], os.environ['PRIVATE_KEY'])
        response = sess.get(fn, verify = sess.verify, stream=True).result() # final boss

        with open(f'/root/csctiming/tmp{idx}.root', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        del response

        try:
            nonlocal hits_bx0
            nonlocal hits_bxneg1
            nonlocal hits_bxpos1
            run_num = runs[idx]
            print(f'Run: {run_num}')
            file = uproot.open(f'/root/csctiming/tmp{idx}.root')
            hits_bx0 += file[f'DQMData/Run {run_num}/L1T/Run summary/L1TStage2EMTF/Timing/cscLCtTimingBX0;1'].values()
            hits_bxneg1 += file[f'DQMData/Run {run_num}/L1T/Run summary/L1TStage2EMTF/Timing/cscLCtTimingBXNeg1;1'].values()
            hits_bxpos1 += file[f'DQMData/Run {run_num}/L1T/Run summary/L1TStage2EMTF/Timing/cscLCtTimingBXPos1;1'].values()
            del file
            file = ROOT.TFile(f'/root/csctiming/tmp{idx}.root')
            for plot_type in plot_types:
                nonlocal plots
                if plot_type == '':
                    continue
                plot_path = f'DQMData/Run {run_num}/{plot_type}'
                ROOT.gStyle.SetOptStat(0)
                ROOT.gRoot.ForceStyle()
                new_plot = file.Get(plot_path)
                new_plot.SetDirectory(0)
                new_plot.Draw()
                new_plot.SetTitle(new_plot.GetTitle()+', Run #'+str(run_num)+', Date: '+str(dates[idx]))
                new_plot.SetName(plot_type+'/'+str(run_num))
                print(plot_type)

                plots.append(new_plot)
            del file
            final_runs += [runs[idx]]
        except Exception as e: print(e)

        os.remove(f'/root/csctiming/tmp{idx}.root')

    
    pool = ThreadPool().imap_unordered(lambda p: process_file(*p), enumerate(file_names))

    for result in pool:
        print(result)
    outfile = ROOT.TFile('/root/csctiming/data.root', 'recreate')

    for plot in plots:
        plot.Write()
        del plot

    del outfile


    # this can definitely be cleaned up but i'm glad it's fully written out for my sake lol
    hits_bx0_noneighbors = np.delete(hits_bx0, [2, 9, 16, 23, 30, 37], 0)
    hits_bxneg1_noneighbors = np.delete(hits_bxneg1, [2, 9, 16, 23, 30, 37], 0)
    hits_bxpos1_noneighbors = np.delete(hits_bxpos1, [2, 9, 16, 23, 30, 37], 0)

    arr_hits_bx0 = np.reshape(hits_bx0_noneighbors, 720, order='F')
    arr_hits_bxneg1 = np.reshape(hits_bxneg1_noneighbors, 720, order='F')
    arr_hits_bxpos1 = np.reshape(hits_bxpos1_noneighbors, 720, order='F')


    station_ring = ['ME-4/2','ME-4/1','ME-3/2','ME-3/1','ME-2/2','ME-2/1','ME-1/3','ME-1/2','ME-1/1b','ME-1/1a','ME+1/1a','ME+1/1b','ME+1/2','ME+1/3','ME+2/1','ME+2/2','ME+3/1','ME+3/2','ME+4/1','ME+4/2']
    inner_station_ring = ['ME-4/1','ME-3/1','ME-2/1','ME+2/1','ME+3/1','ME+4/1']
    clist_ints = []
    for x in range(1,37): clist_ints.append(x)
    chamber = [str(f) for f in clist_ints]

    all_names = []

    for idx_station_ring, station_ring_name in enumerate(station_ring):
        for idx_chamber, chamber_number in enumerate(chamber):
            if station_ring_name in inner_station_ring:
                half_chamber_number = str(int(chamber_number)/2)
                new_name = station_ring_name + '/' + half_chamber_number
            else:
                new_name = station_ring_name + '/' + chamber_number
            all_names.append(new_name)

    df = pd.DataFrame({'Chamber': all_names,
                       'BX-1': arr_hits_bxneg1,
                       'BX0': arr_hits_bx0,
                       'BX+1':arr_hits_bxpos1})

    df_drop_half = (df
            .assign(has_half = lambda x: x['Chamber'].str.contains('/.5'),
                    a_or_b = lambda x: x['Chamber'].str.contains('a') | x['Chamber'].str.contains('b'))
            .query('(~has_half) & (~a_or_b)')
            .drop(['has_half', 'a_or_b'], axis=1))

    subset = (df_drop_half
            .assign(has_point = lambda x: x['Chamber'].str.contains('\.'))
            .query('has_point')
            .assign(Chamber = lambda x: x['Chamber'].str.replace('\.0',''))
            .assign(new_bx1 = lambda x: 2 * x['BX-1'],
                    new_bx0 = lambda x: 2 * x['BX0'],
                    new_bxp1 = lambda x: 2 * x['BX+1'])
            [['Chamber','new_bx1','new_bx0','new_bxp1']]
            .rename({'new_bx1': 'BX-1','new_bx0':'BX0','new_bxp1':'BX+1'}, axis=1))

    df_drop_half.loc[subset.index] = subset


    #create df with ME1/1a chambers
    df1 = df[df['Chamber'].str.contains('a')]
    #get names of ME1 chambers
    me1_names = df1['Chamber'].str.replace('a','',regex=True)
    #create df with ME1/1b chambers
    df2 = df[df['Chamber'].str.contains('b')]
    
    # combine ME1/1a and ME1/1b chambers
    me1_bx0 = df1['BX0'].to_numpy() + df2['BX0'].to_numpy()
    me1_bxneg1 = df1['BX-1'].to_numpy() + df2['BX-1'].to_numpy()
    me1_bxpos1 = df1['BX+1'].to_numpy() + df2['BX+1'].to_numpy()

    df_me1 = pd.DataFrame({'Chamber': me1_names,
                           'BX-1': me1_bxneg1,
                           'BX0': me1_bx0,
                           'BX+1': me1_bxpos1})

    df_final = pd.concat([df_drop_half,df_me1])
    del df_drop_half
    del df_me1
    print(df_final)
    html_df = df_final.to_html()
    del df_final
    return html_df, ','.join(final_runs)
