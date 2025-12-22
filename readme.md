Exploring the Heterogeneity of Tabular Data: A Diversity-aware Data Generator via LLMs

DATE is a LLM-based diversity-aware tabular data generator.

运行ModelShare_with_DSR_final.py, 划分异构数据为block/[dataset_name].csv文件，并存储DGR-data pair在current_block/[dataset_name].csv

运行dataGenerator.py,自动化调用利用LLM实现生成


test_UCB是利用多臂老虎机算法的子集抽取算法，在bank-marketing的modelID=2上选择出了最好的子集（和贪心一致）
test_forward:由前向后贪心，不能保证每次贪心都能找到最优，但是理论上看越靠后越有优势，该算法略优于从后向前贪心
test_leakCheck: 用test挑选最好的子集，有信息泄露。此方法是用于查看哪些子集是最佳，设计方法尽可能接近
test_backward_greedy: 由后向前贪心
