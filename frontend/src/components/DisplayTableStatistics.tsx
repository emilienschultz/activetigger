import { FC } from 'react';
import { MLStatisticsModel } from '../types';

export interface DisplayTableStatisticsProps {
  scores: MLStatisticsModel;
  title?: string | null;
}

interface TableModel {
  index: string[];
  columns: string[];
  data: [number, number, number, number, number][];
}

export const DisplayTableStatistics: FC<DisplayTableStatisticsProps> = ({ scores, title }) => {
  const table = scores.table ? (scores.table as unknown as TableModel) : null;

  const labels = Object.keys(scores['f1_label'] || []);
  const rowCount = table?.data.length || 0;
  const colCount = table?.columns.length || 0;

  return (
    <div className="overflow-x-auto">
      {table && (
        <table className="table-auto border-collapse border border-gray-300 w-full text-sm">
          {title && (
            <caption className="caption-top text-lg font-medium mb-2 text-gray-700">
              {title}
            </caption>
          )}
          <thead>
            <tr>
              <th></th>
              <th></th>
              <td colSpan={labels.length} className="bg-gray-300 text-center px-4 py-2">
                Predicted
              </td>
            </tr>
            <tr className="bg-gray-100">
              <th></th>
              <td className="px-4 py-2">Label</td>
              {table?.columns.map((col, index) => (
                <td key={col} className="px-4 py-2 capitalize font-semibold">
                  {index < labels.length ? <b>{col}</b> : col}
                </td>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.data.map((row, rowIndex) => (
              <tr
                key={table.index[rowIndex]}
                className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
              >
                {rowIndex === 0 && (
                  <td
                    rowSpan={colCount}
                    className="bg-blue-100 font-semibold text-center px-4 py-2"
                  >
                    Truth
                  </td>
                )}
                {rowIndex === colCount && (
                  <td
                    rowSpan={rowCount - colCount}
                    className="bg-green-100 font-semibold text-center px-4 py-2"
                  >
                    Metrics
                  </td>
                )}
                <td className="font-medium px-4 py-2">{table.index[rowIndex]}</td>
                {row.map((cell, colIndex) => (
                  <td key={colIndex} className="px-4 py-2 text-center">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
