import { omit } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';

//import { stringify } from 'csv-stringify/browser/esm/sync';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { useAddProjectFile, useCreateProject } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';
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

  const maxTrainSet = 100000;
  const langages = [
    { value: 'en', label: 'English' },
    { value: 'fr', label: 'French' },
    { value: 'de', label: 'German' },
    { value: 'cn', label: 'Chinese' },
  ];
  const { register, control, handleSubmit, setValue } = useForm<ProjectModel & { files: FileList }>(
    {
      defaultValues: {
        //        project_name: 'New project',
        n_train: 100,
        n_test: 0,
        language: 'en',
        clear_test: false,
        random_selection: true,
      },
    },
  );
  const { notify } = useNotifications();

  const [spinner, setSpinner] = useState<boolean>(false); // state for the data
  const [data, setData] = useState<DataType | null>(null); // state for the data
  const navigate = useNavigate(); // rooting
  const createProject = useCreateProject(); // API call
  const addProjectFile = useAddProjectFile(); // API call
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
      loadFile(file).then((data) => {
        if (data === null) {
          notify({ type: 'error', message: 'Error reading the file' });
          return;
        }
        setData(data);
        setValue('n_train', Math.min(data?.data.length || 0, 100));
      });
    }
  }, [files, maxSize, notify, setValue]);

  // action when form validated
  const onSubmit: SubmitHandler<ProjectModel & { files: FileList }> = async (formData) => {
    if (data) {
      if (formData.col_id == '') {
        notify({ type: 'error', message: 'Please select a id column' });
        return;
      }
      if (!formData.cols_text) {
        notify({ type: 'error', message: 'Please select a text column' });
        return;
      }
      if (Number(formData.n_train) + Number(formData.n_test) > data.data.length) {
        notify({
          type: 'warning',
          message: 'The sum of train and test set is too big, the train set is set to N - testset',
        });
        setValue('n_train', Math.max(0, data.data.length - Number(formData.n_test) - 1));
      }
      setSpinner(true);
      try {
        try {
          // send the data
          await addProjectFile(files[0], formData.project_name);
          // create the project
          const slug = await createProject({
            ...omit(formData, 'files'),
            filename: data.filename,
          });
          setSpinner(false);
          navigate(`/projects/${slug}`);
        } catch (error) {
          notify({ type: 'error', message: error + '' });
          setSpinner(false);
        }
      } catch (error) {
        notify({ type: 'error', message: 'Error creating project' });
        navigate('/projects');
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
              Upload a file in tabular format (csv (; or tab), xlsx or parquet, size limit{' '}
              {maxSizeMo} Mo)
            </li>
            <li>Indicate the columns for id, text</li>
            <li>
              Optional : annotation column & number of elements in testset (not annotated rows)
            </li>
            <li>Validate to create</li>
          </ul>
          <div className="alert alert-warning" role="alert">
            Both project name and index will be modified to be compatible with URLs (slugify). For
            instance, '_' and ' ' will be replaced by '-'. Please be careful, especially if you need
            to later join the data with other tables.
          </div>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="form-frame">
          <div>
            <label className="form-label" htmlFor="project_name">
              Project name
            </label>
            <input
              className="form-control"
              id="project_name"
              placeholder="Name of the project (need to be unique in the system)"
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
                  <div className="m-3">
                    Size of the dataset : <b>{data.data.length - 1}</b>
                  </div>
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
                    Id column (they need to be unique)
                  </label>
                  <select
                    className="form-control"
                    id="col_id"
                    disabled={data === null}
                    {...register('col_id')}
                  >
                    <option key="row_number" value="row_number">
                      Row number
                    </option>
                    {columns}
                  </select>
                </div>
                <div>
                  <label className="form-label" htmlFor="cols_text">
                    Text columns (all the selected fields will be concatenated)
                  </label>

                  <Controller
                    name="cols_text"
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
                    {langages.map((lang) => (
                      <option key={lang.value} value={lang.value}>
                        {lang.label}
                      </option>
                    ))}
                  </select>

                  <label className="form-label" htmlFor="col_label">
                    Columns for existing annotations (optional)
                  </label>
                  <Controller
                    name="cols_label"
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
                  {/* <select
                    className="event-control"
                    id="col_label"
                    disabled={data === null}
                    {...register('col_label')}
                  >
                    <option key="none" value={''} style={{ color: 'grey' }}>
                      Select...
                    </option>
                    {columns}
                  </select> */}

                  <label className="form-label" htmlFor="cols_context">
                    Contextual information columns (optional)
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
                    Number of elements in the train set (limit : 100.000)
                  </label>
                  <input
                    className="form-control"
                    id="n_train"
                    type="number"
                    {...register('n_train')}
                    max={maxTrainSet}
                  />

                  <label className="form-label" htmlFor="n_test">
                    Number of elements in the test set (not already annotated)
                  </label>
                  <input
                    className="form-control"
                    id="n_test"
                    type="number"
                    {...register('n_test')}
                  />

                  <label className="form-label" htmlFor="clear_test">
                    Empty testset{' '}
                    <input
                      id="clear_test"
                      type="checkbox"
                      {...register('clear_test')}
                      className="mx-3"
                    />
                  </label>

                  <label className="form-label" htmlFor="random_selection">
                    Random selection of elements{' '}
                    <a className="randomselection">
                      <HiOutlineQuestionMarkCircle />
                    </a>
                    <Tooltip anchorSelect=".randomselection" place="top">
                      If not, will keep the order (minus empty elements) only if testset = 0
                    </Tooltip>
                    <input
                      id="random_selection"
                      type="checkbox"
                      {...register('random_selection')}
                      className="mx-3"
                    />
                  </label>

                  <label className="form-label" htmlFor="cols_test">
                    Stratify the test set by
                    <a className="stratify">
                      <HiOutlineQuestionMarkCircle />
                    </a>
                    <Tooltip anchorSelect=".stratify" place="top">
                      If not empty, will stratify the test set by the selected column (try to
                      equilibrate the number of elements regarding each category)
                    </Tooltip>
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
