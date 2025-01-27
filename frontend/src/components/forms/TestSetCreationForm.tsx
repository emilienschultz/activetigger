//import { omit } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';

import { omit } from 'lodash';
import { unparse } from 'papaparse';
import { useCreateTestSet } from '../../core/api';
import { loadCSVFile, loadParquetFile } from '../../core/utils';
import { TestSetModel } from '../../types';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

// component
export const TestSetCreationForm: FC<{ projectSlug: string }> = ({ projectSlug }) => {
  // form management
  const { register, control, handleSubmit } = useForm<TestSetModel & { files: FileList }>({
    defaultValues: {},
  });
  const createTestSet = useCreateTestSet(); // API call

  const [data, setData] = useState<DataType | null>(null);
  //const createTestSet = useCreateTestSet();
  const files = useWatch({ control, name: 'files' });
  // available columns
  const columns = data?.headers.map((h) => (
    <option key={h} value={h}>
      {h}
    </option>
  ));

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
    console.log('checking file', files);
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.includes('csv')) {
        console.log('csv');
        loadCSVFile(file).then((data) => {
          setData(data);
        });
      }
      if (file.name.includes('parquet')) {
        console.log('parquet');
        loadParquetFile(file).then((data) => {
          setData(data);
        });
      }
    }
  }, [files]);

  // action when form validated
  const onSubmit: SubmitHandler<TestSetModel & { files: FileList }> = async (formData) => {
    if (data) {
      const csv = data ? unparse(data.data, { header: true, columns: data.headers }) : '';
      await createTestSet(projectSlug, {
        ...omit(formData, 'files'),
        csv,
        filename: data.filename,
      });
    }
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <form onSubmit={handleSubmit(onSubmit)} className="form-frame">
          <div className="explanations">
            No test data set has been created. Do you want to upload your own test set? Be careful,
            if you upload a testset, its id will be modified with "imported_". You need to take care
            of the coherence for the labels.
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
                    Column for text
                  </label>
                  <select
                    className="form-control"
                    id="col_text"
                    disabled={data === null}
                    {...register('col_text')}
                  >
                    <option key="none"></option>

                    {columns}
                  </select>
                  <label className="form-label" htmlFor="col_label">
                    Column for label (optional)
                  </label>
                  <select
                    className="form-control"
                    id="col_label"
                    disabled={data === null}
                    {...register('col_label')}
                  >
                    <option key="none">No label to import</option>

                    {columns}
                  </select>
                  <label className="form-label" htmlFor="n_test">
                    Number of elements
                  </label>
                  <input
                    className="form-control"
                    id="n_test"
                    type="number"
                    {...register('n_test')}
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
  );
};
