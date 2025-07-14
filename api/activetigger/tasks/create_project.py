import shutil

import pandas as pd

from activetigger.datamodels import ProjectBaseModel, ProjectModel
from activetigger.functions import slugify
from activetigger.tasks.base_task import BaseTask


class CreateProject(BaseTask):
    """
    Create a new project
    """

    kind = "create_project"

    def __init__(
        self,
        project_slug: str,
        params: ProjectBaseModel,
        username: str,
        data_all: str = "data_all.parquet",
        train_file: str = "train.parquet",
        test_file: str = "test.parquet",
        features_file: str = "features.parquet",
    ):
        super().__init__()
        self.project_slug = project_slug
        self.params = params
        self.username = username
        self.data_all = data_all
        self.train_file = train_file
        self.test_file = test_file
        self.features_file = features_file

    def __call__(self) -> tuple[ProjectModel, pd.DataFrame | None, pd.DataFrame | None]:
        """
        Create the project with the given name and file
        """
        print(f"Start queue project {self.project_slug} for {self.username}")
        # check if the directory already exists + file (should with the data)
        if self.params.dir is None or not self.params.dir.exists():
            print("The directory does not exist and should", self.params.dir)
            raise Exception("The directory does not exist and should")
        print("Directory exists, continue")
        # Step 1 : load all data and index to str and rename columns
        file_path = self.params.dir.joinpath(self.params.filename)
        if self.params.filename.endswith(".csv"):
            content = pd.read_csv(file_path, low_memory=False)
        elif self.params.filename.endswith(".parquet"):
            content = pd.read_parquet(file_path)
        elif self.params.filename.endswith(".xlsx"):
            content = pd.read_excel(file_path)
        else:
            raise Exception("File format not supported (only csv, xlsx and parquet)")
        # rename columns both for data & params to avoid confusion
        content.columns = ["dataset_" + i for i in content.columns]  # type: ignore[assignment]
        if self.params.col_id:
            self.params.col_id = "dataset_" + self.params.col_id
        # change also the name in the parameters
        self.params.cols_text = ["dataset_" + i for i in self.params.cols_text if i]
        self.params.cols_context = ["dataset_" + i for i in self.params.cols_context if i]
        self.params.cols_label = ["dataset_" + i for i in self.params.cols_label if i]
        self.params.cols_stratify = ["dataset_" + i for i in self.params.cols_stratify if i]

        # remove completely empty lines
        content = content.dropna(how="all")
        all_columns = list(content.columns)
        n_total = len(content)

        # test if the size of the sample requested is possible
        if len(content) < self.params.n_test + self.params.n_train:
            shutil.rmtree(self.params.dir)
            raise Exception(
                f"Not enough data for creating the train/test dataset. Current : {len(content)} ; Selected : {self.params.n_test + self.params.n_train}"
            )

        # create the index
        keep_id = []  # keep unchanged the index to avoid desindexing

        # case of the index should be the row number
        if self.params.col_id == "dataset_row_number":
            print("Use the row number as index")
            content["id"] = [str(i) for i in range(len(content))]
            content.set_index("id", inplace=True)
        # case of a column as index
        else:
            # check if index after slugify is unique otherwise throw an error
            if not (
                (content[self.params.col_id].astype(str).apply(slugify)).nunique() == len(content)
            ):
                shutil.rmtree(self.params.dir)
                raise Exception("The id column is not unique after slugify, please change it")
            content["id"] = content[self.params.col_id].astype(str).apply(slugify)
            keep_id.append(self.params.col_id)
            content.set_index("id", inplace=True)

        # convert columns that can be numeric or force text, exception for the text/labels
        for col in [i for i in content.columns if i not in self.params.cols_label]:
            try:
                content[col] = pd.to_numeric(content[col], errors="raise")
            except Exception:
                content[col] = content[col].astype(str).replace("nan", None)
        for col in self.params.cols_label:
            try:
                content[col] = content[col].astype(str).replace("nan", None)
            except Exception:
                # if the column is not convertible to string, keep it as is
                pass

        # create the text column, merging the different columns
        content["text"] = content[self.params.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )

        # convert NA texts in empty string
        content["text"] = content["text"].fillna("")

        # limit of usable text (in the futur, will be defined by the number of token)
        def limit(text):
            return 1200

        content["limit"] = content["text"].apply(limit)

        # save a complete copy of the dataset
        content.to_parquet(self.params.dir.joinpath(self.data_all), index=True)

        # ------------------------
        # End of the data cleaning
        # ------------------------
        # Step 2 : test dataset : from the complete dataset + random/stratification
        rows_test = []
        self.params.test = False
        testset = None
        if self.params.n_test != 0:
            # if no stratification
            if len(self.params.cols_stratify) == 0:
                testset = content.sample(self.params.n_test)
            # if stratification, total cat, number of element per cat, sample with a lim
            else:
                df_grouped = content.groupby(self.params.cols_stratify, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(self.params.n_test / nb_cat)
                testset = df_grouped.apply(lambda x: x.sample(min(len(x), nb_elements_cat)))
            # save the testset
            testset.to_parquet(self.params.dir.joinpath(self.test_file), index=True)
            self.params.test = True
            rows_test = list(testset.index)

        # Step 3 : train dataset / different strategies

        # remove test rows
        content = content.drop(rows_test)

        # case where there is no test set and the selection is deterministic
        if not self.params.random_selection and self.params.n_test == 0:
            trainset = content[0 : self.params.n_train]
        # case to force the max of label from one column
        elif self.params.force_label and len(self.params.cols_label) > 0:
            f_notna = content[self.params.cols_label[0]].notna()
            f_na = content[self.params.cols_label[0]].isna()
            # different case regarding the number of labels
            if f_notna.sum() > self.params.n_train:
                trainset = content[f_notna].sample(self.params.n_train)
            else:
                n_train_random = self.params.n_train - f_notna.sum()
                trainset = pd.concat([content[f_notna], content[f_na].sample(n_train_random)])
        # case there is stratification on the trainset
        elif len(self.params.cols_stratify) > 0 and self.params.stratify_train:
            df_grouped = content.groupby(self.params.cols_stratify, group_keys=False)
            nb_cat = len(df_grouped)
            nb_elements_cat = round(self.params.n_train / nb_cat)
            trainset = df_grouped.apply(lambda x: x.sample(min(len(x), nb_elements_cat)))
        # default with random selection in the remaining elements
        else:
            print("random selection of the trainset")
            trainset = content.sample(self.params.n_train)

        # write the trainset
        trainset[list(set(["text", "limit"] + self.params.cols_context + keep_id))].to_parquet(
            self.params.dir.joinpath(self.train_file), index=True
        )
        trainset[[]].to_parquet(self.params.dir.joinpath(self.features_file), index=True)

        # save parameters (without the data)
        # params.cols_label = []  # reverse dummy
        project = self.params.model_dump()

        # add elements for the parameters
        project["project_slug"] = self.project_slug
        project["all_columns"] = all_columns
        project["n_total"] = n_total

        # schemes/labels to import (in the main process)
        import_trainset = None
        import_testset = None
        if len(self.params.cols_label) > 0:
            import_trainset = trainset[self.params.cols_label].dropna(how="all")
            if testset is not None and not self.params.clear_test:
                import_testset = testset[self.params.cols_label].dropna(how="all")

        # delete the initial file
        self.params.dir.joinpath(self.params.filename).unlink()

        print(f"End project queue {self.project_slug} created for {self.username}")
        return ProjectModel(**project), import_trainset, import_testset
