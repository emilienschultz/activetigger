import { parquetMetadataAsync, parquetRead } from 'hyparquet';
import { fromPairs, zip } from 'lodash';

import { DataType } from '../components/forms/ProjectCreationForm';

export async function loadParquetFile(file: File): Promise<DataType> {
  const arrayBuffer = await file.arrayBuffer();
  const metadata = await parquetMetadataAsync(arrayBuffer);
  return new Promise((resolve) =>
    parquetRead({
      metadata,
      file: arrayBuffer,
      onComplete: (arrayData) => {
        const headers = metadata.schema.slice(1).map((s) => s.name);
        console.log(metadata.schema);
        const data = arrayData.map((ad) => fromPairs(zip(headers, ad)));
        resolve({ data, headers });
      },
    }),
  );
}
