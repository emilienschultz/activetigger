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
        valid_file: str = "valid.parquet",
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
        self.valid_file = valid_file
        self.features_file = features_file

    def __call__(self) -> tuple[ProjectModel, pd.DataFrame | None, pd.DataFrame | None]:
        """
        Create the project with the given name and file, return the project model
        and the train/test schemes to import in the database
        """
        print(f"Start queue project {self.project_slug} for {self.username}")
        # check if the directory already exists + file (should with the data)
        if self.params.dir is None or not self.params.dir.exists():
            print("The directory does not exist and should", self.params.dir)
            raise Exception("The directory does not exist and should")

        # Step 1 : load all data and index to str and rename columns
        # If file comes from a previous project, there is no need to reprocess it
        if self.params.filename:
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
            if len(content) < self.params.n_test + self.params.n_valid + self.params.n_train:
                shutil.rmtree(self.params.dir)
                raise Exception(
                    f"Not enough data for creating the train/valid/test dataset. Current : {len(content)} ; Selected : {self.params.n_test + self.params.n_valid + self.params.n_train}"
                )

            # create the index
            keep_id = []  # keep unchanged the index to avoid desindexing

            # case there is a id column that is unique
            if self.params.col_id != "dataset_row_number" and (
                (content[self.params.col_id].astype(str).apply(slugify)).nunique() == len(content)
            ):
                content["id"] = content[self.params.col_id].astype(str).apply(slugify)
                keep_id.append(self.params.col_id)
                content.set_index("id", inplace=True)
            # by default the row number
            else:
                print("Use the row number as index")
                content["id"] = [str(i) for i in range(len(content))]
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
        else:
            # case the file is already processed (coming from another project)
            content = pd.read_parquet(self.params.dir.joinpath(self.data_all))
            all_columns = list(content.columns)
            n_total = len(content)
            keep_id = []
            # TODO ATTENTION A L'INDEX

        # ------------------------
        # End of the data cleaning
        # ------------------------
        # Step 2 : valid/test dataset : from the complete dataset + random/stratification
        # if both, draw together valid / test set and then separage
        rows_test = []
        rows_valid = []
        self.params.test = False
        self.params.valid = False
        testset = None
        validset = None
        if self.params.n_test + self.params.n_valid != 0:
            n_to_draw = self.params.n_test + self.params.n_valid
            # if no stratification
            if len(self.params.cols_stratify) == 0:
                draw = content.sample(n_to_draw, random_state=42)
            # if stratification, total cat, number of element per cat, sample with a lim
            else:
                df_grouped = content.groupby(self.params.cols_stratify, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(n_to_draw / nb_cat)
                draw = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat), random_state=42)
                )

            # divide between test and valid
            if self.params.test > 0 and self.params.valid == 0:
                testset = draw
                testset.to_parquet(self.params.dir.joinpath(self.test_file), index=True)
                self.params.test = True
                rows_test = list(testset.index)
            elif self.params.valid > 0 and self.params.test == 0:
                validset = draw
                validset.to_parquet(self.params.dir.joinpath(self.valid_file), index=True)
                self.params.valid = True
                rows_valid = list(validset.index)
            else:
                testset = draw.sample(self.params.n_test, random_state=42)
                validset = draw.drop(index=testset.index)
                validset.to_parquet(self.params.dir.joinpath(self.valid_file), index=True)
                testset.to_parquet(self.params.dir.joinpath(self.test_file), index=True)
                self.params.test = True
                self.params.valid = True
                rows_valid = list(validset.index)
                rows_test = list(testset.index)

        # Step 3 : train dataset / different strategies

        # remove test rows
        content = content.drop(rows_test)
        content = content.drop(rows_valid)

        # case where there is no test set and the selection is deterministic
        if (
            not self.params.random_selection
            and self.params.n_test == 0
            and self.params.n_valid == 0
        ):
            trainset = content[0 : self.params.n_train]
        # case to force the max of label from one column
        elif self.params.force_label and len(self.params.cols_label) > 0:
            f_notna = content[self.params.cols_label[0]].notna()
            f_na = content[self.params.cols_label[0]].isna()
            # different case regarding the number of labels
            if f_notna.sum() > self.params.n_train:
                trainset = content[f_notna].sample(self.params.n_train, random_state=42)
            else:
                n_train_random = self.params.n_train - f_notna.sum()
                trainset = pd.concat(
                    [content[f_notna], content[f_na].sample(n_train_random, random_state=42)]
                )
        # case there is stratification on the trainset
        elif len(self.params.cols_stratify) > 0 and self.params.stratify_train:
            df_grouped = content.groupby(self.params.cols_stratify, group_keys=False)
            nb_cat = len(df_grouped)
            nb_elements_cat = round(self.params.n_train / nb_cat)
            trainset = df_grouped.apply(
                lambda x: x.sample(min(len(x), nb_elements_cat)), random_state=42
            )
        # default with random selection in the remaining elements
        else:
            print("random selection of the trainset")
            trainset = content.sample(self.params.n_train, random_state=42)

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
        import_validset = None
        if len(self.params.cols_label) > 0:
            import_trainset = trainset[self.params.cols_label].dropna(how="all")
            if testset is not None and not self.params.clear_test:
                import_testset = testset[self.params.cols_label].dropna(how="all")
            if validset is not None and not self.params.clear_valid:
                import_validset = validset[self.params.cols_label].dropna(how="all")

        # delete the initial file
        if self.params.filename:
            self.params.dir.joinpath(self.params.filename).unlink()

        return ProjectModel(**project), import_trainset, import_validset, import_testset
