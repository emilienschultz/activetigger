import { omit } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';
import ClipLoader from 'react-spinners/ClipLoader';

//import { stringify } from 'csv-stringify/browser/esm/sync';
import { CanceledError } from 'axios';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { useAddProjectFile, useCreateProject, useProjectNameAvailable } from '../../core/api';
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
    { value: 'ja', label: 'Japanese' },
  ];
  const { register, control, handleSubmit, setValue } = useForm<ProjectModel & { files: FileList }>(
    {
      defaultValues: {
        n_train: 100,
        n_test: 0,
        language: 'en',
        clear_test: false,
        random_selection: true,
        force_label: false,
      },
    },
  );
  const { notify } = useNotifications();

  const [creatingProject, setCreatingProject] = useState<boolean>(false); // state for the data
  const [data, setData] = useState<DataType | null>(null); // state for the data
  const navigate = useNavigate(); // rooting
  const createProject = useCreateProject(); // API call
  const availableProjectName = useProjectNameAvailable(); // check if the project name is available
  const { addProjectFile, progression, cancel } = useAddProjectFile(); // API call
  const files = useWatch({ control, name: 'files' }); // watch the files entry
  const force_label = useWatch({ control, name: 'force_label' }); // watch the force label entry

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
          message: `File is too big (only file less than ${maxSizeMo} Mo are allowed)`,
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
      // check the form
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
      setCreatingProject(true);

      try {
        // test if the project name is available
        const available = await availableProjectName(formData.project_name);
        if (formData.project_name === '') {
          notify({ type: 'error', message: 'Please select a project name' });
          setCreatingProject(false);
          return;
        }
        if (!available) {
          notify({ type: 'error', message: 'Project name already taken' });
          setCreatingProject(false);
          return;
        }
        // send the data
        await addProjectFile(files[0], formData.project_name);
        console.log('file uploaded');
        // create the project
        const slug = await createProject({
          ...omit(formData, 'files'),
          filename: data.filename,
        });
        setCreatingProject(false);
        navigate(`/projects/${slug}`);
      } catch (error) {
        // if error comes from axios being canceled by user than show a success else that's a real error
        if (!(error instanceof CanceledError)) notify({ type: 'error', message: error + '' });
        else notify({ type: 'success', message: 'Project creation aborted' });
      } finally {
        // anyway stop the loading state
        setCreatingProject(false);
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

        <form onSubmit={handleSubmit(onSubmit)} className="form-frame ">
          <div className=" position-relative">
            <div>
              <label className="form-label" htmlFor="project_name">
                Project name
              </label>
              <input
                className="form-control"
                id="project_name"
                placeholder="Name of the project (need to be unique in the system)"
                type="text"
                disabled={creatingProject}
                {...register('project_name')}
                onClick={handleClickOnText}
              />
            </div>

            <div>
              <label className="form-label" htmlFor="csvFile">
                Data
              </label>
              <input
                className="form-control"
                disabled={creatingProject}
                id="csvFile"
                type="file"
                {...register('files')}
              />
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
                        data.data
                          .slice(0, 5)
                          .map((row) =>
                            Object.fromEntries(
                              Object.entries(row).map(([key, value]) => [key, String(value)]),
                            ),
                          ) as Record<keyof DataType['headers'], string>[]
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
                      disabled={data === null || creatingProject}
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
                          isDisabled={creatingProject}
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
                      disabled={data === null || creatingProject}
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
                          isDisabled={creatingProject}
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
                          isDisabled={creatingProject}
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
                      disabled={creatingProject}
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
                      disabled={creatingProject}
                      {...register('n_test')}
                    />

                    <label className="form-label" htmlFor="force_label">
                      Prioritize existing labels{' '}
                      <a className="force_label">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".force_label" place="top">
                        Select in priority the elements with existing labels (if any) of the first
                        column of labels
                      </Tooltip>
                      <input
                        id="force_label"
                        type="checkbox"
                        disabled={creatingProject}
                        {...register('force_label')}
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
                        disabled={creatingProject || force_label}
                        {...register('random_selection')}
                        className="mx-3"
                      />
                    </label>

                    <label className="form-label" htmlFor="stratify_train">
                      Stratify trainset{' '}
                      <a className="stratify_train">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".stratify_train" place="top">
                        If selected, use the stratify columsn to stratify train set. Small variation
                        in the number of elements can happen.
                      </Tooltip>
                      <input
                        id="stratify_train"
                        type="checkbox"
                        disabled={creatingProject || force_label}
                        {...register('stratify_train')}
                        className="mx-3"
                      />
                    </label>

                    <label className="form-label" htmlFor="stratify_test">
                      Stratify testset{' '}
                      <a className="stratify_train">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".stratify_train" place="top">
                        If selected, use the stratify columsn to stratify test set. Small variation
                        in the number of elements can happen.
                      </Tooltip>
                      <input
                        id="stratify_test"
                        type="checkbox"
                        disabled={creatingProject || force_label}
                        {...register('stratify_test')}
                        className="mx-3"
                      />
                    </label>

                    <label className="form-label" htmlFor="cols_stratify">
                      Columns to stratify
                      <a className="stratify">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".stratify" place="top">
                        If not empty, will stratify by the selected column (try to equilibrate the
                        number of elements regarding each category)
                      </Tooltip>
                    </label>
                    <Controller
                      name="cols_stratify"
                      control={control}
                      render={({ field: { onChange } }) => (
                        <Select
                          options={columnsSelect}
                          isMulti
                          isDisabled={creatingProject}
                          onChange={(selectedOptions) => {
                            onChange(
                              selectedOptions ? selectedOptions.map((option) => option.value) : [],
                            );
                          }}
                        />
                      )}
                    />

                    <label className="form-label" htmlFor="clear_test">
                      Empty testset{' '}
                      <a className="emptytestset">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      <Tooltip anchorSelect=".emptytestset" place="top">
                        Drop labels for the testset
                      </Tooltip>
                      <input
                        id="clear_test"
                        type="checkbox"
                        disabled={creatingProject}
                        {...register('clear_test')}
                        className="mx-3"
                      />
                    </label>
                  </div>
                </div>
              )
            }
            {/* 
              Quasi Modal
              overlay progression bar with cancel button 
            */}
            {data && creatingProject && (
              <div>
                <div className="position-absolute bg-white w-100 h-100 top-0 left-0 d-flex flex-column justify-content-center bg-opacity-50">
                  <div className="d-flex flex-column bg-white p-4 border border-dark gap-2">
                    <div className="d-flex align-items-center gap-2 ">
                      <ClipLoader /> <span>Uploading</span>{' '}
                      <span>
                        {progression.loaded && progression.total
                          ? `${((progression.loaded / progression.total) * 100).toFixed(2)}%`
                          : null}
                      </span>
                    </div>
                    <progress
                      id="upload-progress"
                      value={progression.loaded}
                      max={progression.total}
                    />
                    {cancel !== undefined && (
                      <div>
                        <button
                          className="btn btn-warning mt-1"
                          onClick={() => {
                            cancel.abort();
                          }}
                        >
                          Cancel
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
          {data !== null && (
            <>
              <button
                type="submit"
                className="btn btn-primary form-button"
                disabled={creatingProject}
              >
                Create
              </button>
            </>
          )}
        </form>
      </div>
    </div>
  );
};
