# inDelphi-model
Copyright 2018 MIT, All Rights Reserved.

Requirements: Only standard python packages (numpy, pandas, scipy, collections, os, pickle, copy)

Usage:

>>> import inDelphi
>>> inDelphi.init_model(celltype = user_specified_celltype)

Note: Supported cell types include mESC, U2OS, HEK293, HCT116, and K562. See www.crisprindelphi.design for a user guide on how to choose a cell type for your purposes.

>>> prediction_dataframe, statistics = inDelphi.predict(seq, cutsite)

Note: Sequence is a string of DNA characters. Cutsite specifies the 0-index position of the cutsite, such that seq[:cutsite] and seq[cutsite:] in Python notation describe the cut products.

For assistance on interpreting prediction_dataframe, refer to the Supplementary Information of our manuscript.

Statistics includes the following statistics:
Phi (Microhomology strength)
Precision
1-bp ins frequency
MH del frequency
MHless del frequency
Frameshift frequency
Frame +0 frequency
Frame +1 frequency
Frame +2 frequency
Highest outcome frequency
Highest del frequency
Highest ins frequency
Expected indel length
Reference sequence
Cutsite
gRNA
gRNA orientation
Cas9 type
Celltype
