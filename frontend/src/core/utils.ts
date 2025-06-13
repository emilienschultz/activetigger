import { parquetMetadataAsync, parquetRead } from 'hyparquet';
import { fromPairs, zip } from 'lodash';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';

import { DataType } from '../components/forms/ProjectCreationForm';

/**
 * loadParquetFile
 * parses a parquet file in memory
 * @param file: uploaded file
 * @returns a DataType object storing the data, fieldnames and original filename
 */
export async function loadParquetFile(file: File): Promise<DataType> {
  // since this function is async it must returns a Promise which contains a DataType object

  // read the uploaded file as an arrayBuffer
  const arrayBuffer = await file.arrayBuffer();
  // use th hyparquet library to read the file metadata (which stores the headers)
  const metadata = await parquetMetadataAsync(arrayBuffer);
  // here we have to create a new Promise as the parquetRead function only support the callback syntax
  // The Promise will be return immediately but it will be replaced by what is provided to its resolve function
  // It will be replaced only when the resolve function is called
  // we differ the resolution.
  return new Promise((resolve) =>
    parquetRead({
      metadata,
      file: arrayBuffer,
      // here is the callback which will be called once the file has been loaded
      onComplete: (arrayData) => {
        // extract headers from metadata
        const headers = metadata.schema.slice(1).map((s) => s.name);
        // transforme the data as an array of objects
        const data = arrayData.map((ad) => fromPairs(zip(headers, ad)));
        // resolve the promise with a dataType object
        resolve({ data, headers, filename: file.name });
      },
    }),
  );
}

/**
 * loadCSVFile
 * parses a csv file in memory
 * @param file: uploaded file
 * @returns a DataType object storing the data, fieldnames and original filename
 */
export async function loadCSVFile(file: File): Promise<DataType> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const csvContent = e.target?.result;

      if (typeof csvContent === 'string') {
        // detect if this is coma or tab separated value
        const line = csvContent.split('\n')[0];
        const tabCount = (line.match(/\t/g) || []).length;
        const separator = tabCount > 3 ? '\t' : ',';

        Papa.parse<Record<string, string>>(csvContent, {
          header: true,
          delimiter: separator,
          complete: (results) => {
            const headers = results.meta.fields || [];
            resolve({ data: results.data, headers, filename: file.name });
          },
        });
      } else {
        reject(new Error('Failed to read the CSV file as text.'));
      }
    };

    // Define the onerror callback for any file reading errors
    reader.onerror = () => {
      reject(new Error('Error reading the CSV file.'));
    };

    // Start reading the file as text
    reader.readAsText(file);
  });
}

export async function loadExcelFile(file: File): Promise<DataType> {
  // Read the uploaded file as an arrayBuffer
  const arrayBuffer = await file.arrayBuffer();

  // Use XLSX to read the file as a workbook
  const workbook = XLSX.read(arrayBuffer, { type: 'array' });

  // Assuming we want the first sheet
  const sheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[sheetName];

  // Convert the sheet to JSON format with headers
  const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

  // Extract headers (first row)
  const headers = jsonData[0] as string[];

  // Extract data (subsequent rows)
  const data = jsonData.slice(1).map((row) => fromPairs(zip(headers, row as string[])));

  // Return a DataType object with the data, headers, and filename
  return { data, headers, filename: file.name };
}

// export async function loadFeatherFile(file: File): Promise<DataType> {
//   // Read the uploaded file as an arrayBuffer
//   const arrayBuffer = await file.arrayBuffer();

//   // Use the Apache Arrow library to read the Feather file
//   const table = tableFromIPC(new Uint8Array(arrayBuffer));

//   // Extract headers (column names)
//   const headers = table.schema.fields.map((field) => field.name);

//   // Extract the data as an array of objects
//   const data = [];
//   for (let i = 0; i < table.numRows; i++) {
//     const row = {};
//     headers.forEach((header, colIndex) => {
//       row[header] = table.getColumnAt(colIndex).get(i);
//     });
//     data.push(row);
//   }

//   // Resolve the promise with a DataType object
//   return { data, headers, filename: file.name };
// }

export async function loadFile(file: File): Promise<DataType | null> {
  let data;

  if (file.name.includes('parquet')) {
    console.log('parquet');
    data = await loadParquetFile(file);
  } else if (file.name.includes('csv')) {
    console.log('csv');
    data = await loadCSVFile(file);
  } else if (file.name.includes('xlsx')) {
    console.log('xlsx');
    data = await loadExcelFile(file);
  } else {
    return null;
  }
  return data;
}

/**
 * Display the date to the "from ago" format.
 */
export function dateToFromAgo(date: Date): string {
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  const prefix = seconds < 0 ? 'in ' : '';
  const suffix = seconds < 0 ? '' : ' ago';
  const absSecond = Math.abs(seconds);

  const times = [
    absSecond / 60 / 60 / 24 / 365, // years
    absSecond / 60 / 60 / 24 / 30, // months
    absSecond / 60 / 60 / 24 / 7, // weeks
    absSecond / 60 / 60 / 24, // days
    absSecond / 60 / 60, // hours
    absSecond / 60, // minutes
    absSecond, // seconds
  ];

  return (
    ['year', 'month', 'week', 'day', 'hour', 'minute', 'second']
      .map((name, index) => {
        const time = Math.floor(times[index]);
        if (time > 0) return `${prefix}${time} ${name}${time > 1 ? 's' : ''}${suffix}`;
        return null;
      })
      .reduce((acc, curr) => (acc === null && curr !== null ? curr : null), null) || 'now'
  );
}

// reorder function
export const reorderLabels = (labels: string[], orderedList: string[] | null): string[] => {
  if (!orderedList || orderedList.length === 0) {
    return labels;
  }
  const orderedSet = new Set(orderedList);
  const inOrder: string[] = [];
  const leftovers: string[] = [];
  for (const label of labels) {
    if (orderedSet.has(label)) {
      inOrder.push(label);
    } else {
      leftovers.push(label);
    }
  }
  inOrder.sort((a, b) => orderedList.indexOf(a) - orderedList.indexOf(b));
  return [...inOrder, ...leftovers];
};
