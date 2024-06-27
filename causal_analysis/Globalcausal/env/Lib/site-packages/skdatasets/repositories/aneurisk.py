"""Data from the AneuRisk project."""

import numpy as np
from sklearn.utils import Bunch

from .base import fetch_zip

DESCR = """
The AneuRisk data set is based on a set of three-dimensional angiographic
images taken from 65 subjects, hospitalized at Niguarda Ca’ Granda
Hospital (Milan), who were suspected of being affected by cerebral aneurysms.
Out of these 65 subjects, 33 subjects have an aneurysm at or after the
terminal bifurcation of the ICA (“Upper” group), 25 subjects have an aneurysm
along the ICA (“Lower” group), and 7 subjects were not found any visible
aneurysm during the angiography (“No-aneurysm” group).

For more information see:
http://ecm2.mathcs.emory.edu/aneuriskdata/files/ReadMe_AneuRisk-website_2012-05.pdf
"""


def fetch(name="Aneurisk65", *, data_home=None, return_X_y=False):

    if name != "Aneurisk65":
        raise ValueError(f"Unknown dataset {name}")

    n_samples = 65

    url = "http://ecm2.mathcs.emory.edu/aneuriskdata/files/Carotid-data_MBI_workshop.zip"

    dataset_path = fetch_zip(
        dataname=name,
        urlname=url,
        subfolder="aneurisk",
        data_home=data_home,
    )

    patient_dtype = [
        ('patient', np.int_),
        ('code', 'U8'),
        ('type', 'U1'),
        ('aneurysm location', np.float_),
        ('left_right', 'U2'),
    ]

    functions_dtype = [
        ('curvilinear abscissa', np.object_),
        ('MISR', np.object_),
        ('X0 observed', np.object_),
        ('Y0 observed', np.object_),
        ('Z0 observed', np.object_),
        ('X0 observed FKS', np.object_),
        ('Y0 observed FKS', np.object_),
        ('Z0 observed FKS', np.object_),
        ('X0 observed FKS reflected', np.object_),
        ('X1 observed FKS', np.object_),
        ('Y1 observed FKS', np.object_),
        ('Z1 observed FKS', np.object_),
        ('X1 observed FKS reflected', np.object_),
        ('X2 observed FKS', np.object_),
        ('Y2 observed FKS', np.object_),
        ('Z2 observed FKS', np.object_),
        ('X2 observed FKS reflected', np.object_),
        ('Curvature FKS', np.object_),
    ]

    complete_dtype = patient_dtype + functions_dtype

    X = np.zeros(shape=n_samples, dtype=complete_dtype)

    X[[p[0] for p in patient_dtype]] = np.genfromtxt(
        dataset_path / 'Patients.txt',
        dtype=patient_dtype,
        skip_header=1,
        missing_values=('NA',),
    )

    for i in range(n_samples):
        file = f"Rawdata_FKS_{i + 1}.txt"

        functions = np.genfromtxt(
            dataset_path / file,
            skip_header=1,
        )

        for j, (f_name, _) in enumerate(functions_dtype):
            X[i][f_name] = functions[:, j]

    X = np.array(X.tolist(), dtype=np.object_)

    if return_X_y:
        return X, None

    return Bunch(
        data=X,
        target=None,
        train_indices=[],
        validation_indices=[],
        test_indices=[],
        name=name,
        DESCR=DESCR,
        feature_names=[t[0] for t in complete_dtype],
    )
