import { stringify } from 'csv-stringify/browser/esm/sync';
import { omit } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { usePostAnnotationsFile } from '../core/api';
import { useNotifications } from '../core/notifications';
import { loadFile } from '../core/utils';

interface ImportPropos {
  projectName: string | null;
  currentScheme: string | null;
}

export interface DataSetAnnotations {
  col_id: string;
  col_label: string;
}

export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

export const ImportAnnotations: FC<ImportPropos> = ({ projectName, currentScheme }) => {
  const maxSizeMo = 100;
  const maxSize = maxSizeMo * 1024 * 1024; // 100 MB in bytes
  const { notify } = useNotifications();
  const postAnnotationsFile = usePostAnnotationsFile(projectName || null);

  // form management
  const { register, control, handleSubmit, reset } = useForm<
    DataSetAnnotations & { files: FileList }
  >({
    defaultValues: {},
  });
  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' }); // watch the files entry

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
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
      });
    }
  }, [files, maxSize, notify, setData]);

  // action when form validated
  const onSubmit: SubmitHandler<DataSetAnnotations & { files: FileList }> = async (formData) => {
    if (data) {
      const csv = stringify(data.data, { header: true, columns: data.headers.filter(Boolean) });
      await postAnnotationsFile({
        ...omit(formData, 'files'),
        csv: csv,
        filename: data.filename,
        scheme: currentScheme || '',
      });
      setData(null);
      reset();
    }
  };

  // available columns to display
  const columns = data?.headers.map((h) => (
    <option key={h} value={h}>
      {h}
    </option>
  ));

  return (
    <div className="container">
      <div className="row">
        <h4 className="subsection">Import annotations</h4>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div>
            <div className="alert alert-info">
              You can import annotations for existing elements in the train set. Be sure to have
              exactly the same id for the elements. If elements are already labelled, this
              annotation will prevail. Labels are not checked for existance.
            </div>
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
                    Column for id (they need to match exactly the original data)
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
                  <label className="form-label" htmlFor="col_label">
                    Column for label to import (empty will be droped)
                  </label>
                  <select
                    className="form-control"
                    id="col_label"
                    disabled={data === null}
                    {...register('col_label')}
                  >
                    {columns}
                  </select>
                </div>
                <button type="submit" className="btn btn-primary form-button">
                  Send those annotations
                </button>
              </div>
            )
          }
        </form>
      </div>
    </div>
  );
};
