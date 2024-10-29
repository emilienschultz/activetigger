import { omit } from 'lodash';
import { unparse } from 'papaparse';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';

import { useCreateProject } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadCSVFile, loadExcelFile, loadParquetFile } from '../../core/utils';
import { ProjectModel } from '../../types';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

// component
export const ProjectCreationForm: FC = () => {
  // form management
  const maxSizeMo = 400;
  const maxSize = maxSizeMo * 1024 * 1024; // 100 MB in bytes
  const { register, control, handleSubmit } = useForm<ProjectModel & { files: FileList }>({
    defaultValues: {
      project_name: 'New project',
      n_train: 100,
      n_test: 0,
      language: 'en',
    },
  });
  const { notify } = useNotifications();

  const [spinner, setSpinner] = useState<boolean>(false); // state for the data
  const [data, setData] = useState<DataType | null>(null); // state for the data
  const navigate = useNavigate(); // rooting
  const createProject = useCreateProject(); // API call
  const files = useWatch({ control, name: 'files' }); // watch the files entry
  // available columns
  const columns = data?.headers
    .filter((h) => h !== '')
    .map((h) => (
      <option key={h} value={h}>
        {h}
      </option>
    ));
  const columnsSelect =
    data?.headers.filter((h) => h !== '').map((e) => ({ value: e, label: e })) || [];

  // select the text on input on click
  const handleClickOnText = (event: React.MouseEvent<HTMLInputElement>) => {
    const target = event.target as HTMLInputElement;
    target.select(); // Select the content of the input
  };

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
    console.log('checking file', files);
    if (files && files.length > 0) {
      const file = files[0];
      if (file.size > maxSize) {
        notify({
          type: 'error',
          message: `File is too big (only file less than ${maxSizeMo} are allowed)`,
        });
        return;
      }
      if (file.name.includes('parquet')) {
        console.log('parquet');
        loadParquetFile(file).then((data) => {
          setData(data);
        });
      } else if (file.name.includes('csv')) {
        console.log('csv');
        loadCSVFile(file).then((data) => {
          setData(data);
        });
      } else if (file.name.includes('xlsx')) {
        console.log('csv');
        loadExcelFile(file).then((data) => {
          setData(data);
        });
      } else {
        notify({ type: 'error', message: 'Only csv or parquet files are allowed for now' });
        return;
      }
    }
  }, [files, maxSize, notify]);

  // action when form validated
  const onSubmit: SubmitHandler<ProjectModel & { files: FileList }> = async (formData) => {
    if (data) {
      if (formData.col_id == '') {
        notify({ type: 'error', message: 'Please select a id column' });
        return;
      }
      if (!formData.col_text) {
        notify({ type: 'error', message: 'Please select a text column' });
        return;
      }
      setSpinner(true);
      console.log('ENVOI');
      try {
        // ERROR : problème avec gros parquet unparse marche pas
        const csv = data ? unparse(data.data, { header: true, columns: data.headers }) : '';
        console.log('data parsing done');
        try {
          await createProject({ ...omit(formData, 'files'), csv, filename: data.filename });
          setSpinner(false);
          navigate(`/projects/`);
        } catch (error) {
          setSpinner(false);
        }
      } catch (error) {
        console.log('ERROR WITH UNPARSE');
      }
    }
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="explanations">
          Create a new project.
          <ul>
            <li>
              upload a file in tabular format (csv, xlsx or parquet, size limit {maxSizeMo} Mo)
            </li>
            <li>indicate the columns for id, text, and if needed label & test</li>
            <li>validate to create</li>
          </ul>
        </div>
        <form onSubmit={handleSubmit(onSubmit)} className="form-frame">
          <div>
            <label className="form-label" htmlFor="project_name">
              Project name
            </label>
            <input
              className="form-control"
              id="project_name"
              type="text"
              {...register('project_name')}
              onClick={handleClickOnText}
            />
          </div>

          <div>
            <label className="form-label" htmlFor="csvFile">
              Data
            </label>
            <input className="form-control" id="csvFile" type="file" {...register('files')} />
            {
              // display datable if data available
              data !== null && (
                <div>
                  <div>Preview</div>
                  <DataTable<Record<DataType['headers'][number], string | number>>
                    columns={data.headers.map((h) => ({
                      name: h,
                      selector: (row) => row[h],
                      format: (row) => {
                        const v = row[h];
                        return typeof v === 'bigint' ? Number(v) : v;
                      },
                      width: '200px',
                    }))}
                    data={
                      data.data.slice(0, 5) as Record<keyof DataType['headers'], string | number>[]
                    }
                  />
                </div>
              )
            }
          </div>

          {
            // only display if data
            data != null && (
              <div>
                <div>
                  <label className="form-label" htmlFor="col_id">
                    Column for id (they need to be unique)
                  </label>
                  <select
                    className="form-control"
                    id="col_id"
                    disabled={data === null}
                    {...register('col_id')}
                  >
                    {columns}
                  </select>
                </div>
                <div>
                  <label className="form-label" htmlFor="col_text">
                    Column for text (all the selected fields will be concactenated)
                  </label>

                  <Controller
                    name="col_text"
                    control={control}
                    render={({ field: { onChange } }) => (
                      <Select
                        options={columnsSelect}
                        isMulti
                        onChange={(selectedOptions) => {
                          onChange(
                            selectedOptions ? selectedOptions.map((option) => option.value) : [],
                          );
                        }}
                      />
                    )}
                  />

                  <label className="form-label" htmlFor="language">
                    Language of the corpus (for tokenization and word segmentation)
                  </label>
                  <select
                    className="form-control"
                    id="language"
                    disabled={data === null}
                    {...register('language')}
                  >
                    <option key="en" value={'en'}>
                      English
                    </option>
                    <option key="fr" value={'fr'}>
                      French
                    </option>
                  </select>

                  <label className="form-label" htmlFor="col_label">
                    Column for label (if exists)
                  </label>
                  <select
                    className="form-control"
                    id="col_label"
                    disabled={data === null}
                    {...register('col_label')}
                  >
                    <option key="none"></option>
                    {columns}
                  </select>

                  <label className="form-label" htmlFor="cols_context">
                    Column for contextual information to display (if needed)
                  </label>
                  <Controller
                    name="cols_context"
                    control={control}
                    render={({ field: { onChange } }) => (
                      <Select
                        options={columnsSelect}
                        isMulti
                        onChange={(selectedOptions) => {
                          onChange(
                            selectedOptions ? selectedOptions.map((option) => option.value) : [],
                          );
                        }}
                      />
                    )}
                  />

                  <label className="form-label" htmlFor="n_train">
                    Number of elements in the train set
                  </label>
                  <input
                    className="form-control"
                    id="n_train"
                    type="number"
                    {...register('n_train')}
                  />

                  <label className="form-label" htmlFor="n_test">
                    Number of elements in the test set
                  </label>
                  <input
                    className="form-control"
                    id="n_test"
                    type="number"
                    {...register('n_test')}
                  />

                  <label className="form-label" htmlFor="cols_test">
                    Stratify the test set by
                  </label>
                  <Controller
                    name="cols_test"
                    control={control}
                    render={({ field: { onChange } }) => (
                      <Select
                        options={columnsSelect}
                        isMulti
                        onChange={(selectedOptions) => {
                          onChange(
                            selectedOptions ? selectedOptions.map((option) => option.value) : [],
                          );
                        }}
                      />
                    )}
                  />
                </div>
                <button type="submit" className="btn btn-primary form-button" disabled={spinner}>
                  Create
                </button>
                {spinner && (
                  <div className="col-12 text-center">
                    <PulseLoader />
                  </div>
                )}
              </div>
            )
          }
        </form>
      </div>
    </div>
  );
};
