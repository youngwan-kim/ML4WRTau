from graph_builder import GraphConfig, HEPGraphDatasetBuilder

def main():
    #config = GraphConfig("/data6/Users/youngwan/SKNanoAnalyzer/data/Run3_v12_Run2_v9/2022/Sample/ForSNU/WRtoTauTauJJ_WR3000_N200.json")
    #config = GraphConfig("/data6/Users/youngwan/SKNanoAnalyzer/data/Run3_v12_Run2_v9/2022/Sample/ForSNU/WJets_MG.json")
    '''config = GraphConfig("json/WRtoNTautoTauTauJJ_WR3000_N400_TuneCP5_13p6TeV_madgraph-pythia8.json")
    builder = HEPGraphDatasetBuilder(config, max_workers=25)
    builder.process_files()'''

    for mn in [600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800] :
        config = GraphConfig(f"json/WRtoNTautoTauTauJJ_WR3000_N{mn}_TuneCP5_13p6TeV_madgraph-pythia8.json")
        builder = HEPGraphDatasetBuilder(config, max_workers=10)
        builder.process_files()
    

if __name__ == "__main__":
    main()
