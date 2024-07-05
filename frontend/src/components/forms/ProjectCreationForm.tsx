import { omit } from 'lodash';
import { unparse } from 'papaparse';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';

import { loadParquetFile } from '../../core/utils';
import { ProjectModel } from '../../types';

export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
}

export const ProjectCreationForm: FC = () => {
  const { register, control, handleSubmit } = useForm<ProjectModel & { files: FileList }>();
  const [data, setData] = useState<DataType | null>(null);

  const files = useWatch({ control, name: 'files' });
  useEffect(() => {
    console.log('checking file', files);
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.includes('parquet')) {
        console.log('parquet');
        loadParquetFile(file).then((data) => {
          console.log(data);
          setData(data);
        });
      }
    }
  }, [files]);

  const onSubmit: SubmitHandler<ProjectModel & { files: FileList }> = (formData) => {
    const csv = data ? unparse(data.data, { header: true, columns: data.headers }) : '';
    console.log('new project payload to send to API', { ...omit(formData, 'files'), csv });
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <form onSubmit={handleSubmit(onSubmit)}>
          <div>
            <label className="form-label" htmlFor="project_name">
              Project name
            </label>
            <input
              className="form-control"
              id="project_name"
              type="text"
              {...register('project_name')}
            />
          </div>

          <div>
            <label className="form-label" htmlFor="csvFile"></label>
            <input className="form-control" id="csvFile" type="file" {...register('files')} />
            {data !== null && (
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
                  data.data.slice(0, 10) as Record<keyof DataType['headers'], string | number>[]
                }
              />
            )}
          </div>
          <div>
            <label className="form-label" htmlFor="col_id">
              Column for id
            </label>
            <select
              className="form-control"
              id="col_id"
              disabled={data === null}
              {...register('col_id')}
            >
              {data?.headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              )) || (
                <option value="" disabled selected>
                  First upload a dataset, then select an column here
                </option>
              )}
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
              {data?.headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              )) || (
                <option value="" disabled selected>
                  First upload a dataset, then select an column here
                </option>
              )}
            </select>
          </div>
        </form>
      </div>
    </div>
  );
};
