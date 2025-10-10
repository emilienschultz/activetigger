import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import Select from 'react-select';

import { omit } from 'lodash';
import { unparse } from 'papaparse';
import { useCreateValidSet, useDropEvalSet } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';

import { useNavigate } from 'react-router-dom';
import { EvalSetModel } from '../../types';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

export interface EvalSetsManagementModel {
  projectSlug: string;
  currentScheme: string;
  dataset: string;
  exist: boolean;
}

// component
export const EvalSetsManagement: FC<EvalSetsManagementModel> = ({
  projectSlug,
  currentScheme,
  dataset,
  exist,
}) => {
  // form management
  const { register, control, handleSubmit, setValue } = useForm<EvalSetModel & { files: FileList }>(
    {
      defaultValues: { scheme: currentScheme },
    },
  );

  const createValidSet = useCreateValidSet(); // API call
  const { notify } = useNotifications();

  const dropEvalSet = useDropEvalSet(projectSlug); // API call to drop existing test set
  const navigate = useNavigate(); // for navigation after drop

  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' });
  // available columns
  const columns = data?.headers.map((h) => (
    <option key={`${h}`} value={`${h}`}>
      {h}
    </option>
  ));

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
    console.log('checking file', files);
    if (files && files.length > 0) {
      const file = files[0];
      loadFile(file).then((data) => {
        if (data === null) {
          notify({ type: 'error', message: 'Error reading the file' });
          return;
        }
        setData(data);
        setValue('n_eval', data.data.length - 1);
      });
    }
  }, [files, setValue, notify]);

  // action when form validated
  const onSubmit: SubmitHandler<EvalSetModel & { files: FileList }> = async (formData) => {
    if (data) {
      if (!formData.col_id || !formData.cols_text || !formData.n_eval) {
        notify({ type: 'error', message: 'Please fill all the fields' });
        return;
      }
      const csv = data ? unparse(data.data, { header: true, columns: data.headers }) : '';
      formData.scheme = currentScheme;
      await createValidSet(projectSlug, dataset, {
        ...omit(formData, 'files'),
        csv,
        filename: data.filename,
      });
    }
  };

  return (
    <div className="container">
      <div className="row">
        <div className="col-12">
          {exist && (
            <button
              className="delete-button mt-3"
              onClick={() => {
                dropEvalSet(dataset).then(() => {
                  navigate(0);
                });
              }}
            >
              Drop existing {dataset}
            </button>
          )}
        </div>
      </div>

      {!exist && (
        <div className="row">
          <h4 className="subsection">Import a {dataset} set</h4>
          <div className="alert alert-info">
            <form onSubmit={handleSubmit(onSubmit)}>
              No {dataset} data set has been created. You can upload a {dataset} set. Careful : id
              will be modified with "imported_".
              <label className="form-label" htmlFor="csvFile">
                File to upload
              </label>
              <input className="form-control" id="csvFile" type="file" {...register('files')} />
              {
                // display datable if data available
                data !== null && (
                  <div>
                    <div>Preview</div>
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
                        data.data.slice(0, 5) as Record<
                          keyof DataType['headers'],
                          string | number
                        >[]
                      }
                    />
                  </div>
                )
              }
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
                      <label className="form-label" htmlFor="cols_text">
                        Text columns (all the selected fields will be concatenated)
                      </label>
                      <Controller
                        name="cols_text"
                        control={control}
                        render={({ field: { onChange } }) => (
                          <Select
                            options={(data?.headers || []).map((e) => ({ value: e, label: e }))}
                            isMulti
                            onChange={(selectedOptions) => {
                              onChange(
                                selectedOptions
                                  ? selectedOptions.map((option) => option.value)
                                  : [],
                              );
                            }}
                          />
                        )}
                      />
                      <label className="form-label" htmlFor="col_label">
                        Column for label (optional but they need to exist in the scheme)
                      </label>
                      <select
                        className="form-control"
                        id="col_label"
                        disabled={data === null}
                        {...register('col_label')}
                      >
                        <option key="none" value="">
                          No label
                        </option>

                        {columns}
                      </select>
                      <label className="form-label" htmlFor="n_test">
                        Number of elements
                      </label>
                      <input
                        className="form-control"
                        id="n_test"
                        type="number"
                        {...register('n_eval')}
                      />
                    </div>
                    <button type="submit" className="btn btn-primary form-button">
                      Create
                    </button>
                  </div>
                )
              }
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
