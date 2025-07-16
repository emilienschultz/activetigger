import { FC } from 'react';
import { MLStatisticsModel } from '../types';

export interface DisplayTableStatisticsProps {
  scores: MLStatisticsModel;
  title?: string | null;
}

interface TableModel {
  index: string[];
  columns: string[];
  data: number[][];
}

export const DisplayTableStatistics: FC<DisplayTableStatisticsProps> = ({ scores, title }) => {
  const table = scores.table ? (scores.table as unknown as TableModel) : null;
  const labels = Object.keys(scores['f1_label'] || []);
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
              <td colSpan={labels.length} className="bg-gray-300 text-center p-2">
                Predicted
              </td>
              <td colSpan={3} className="bg-gray-300 text-center p-2">
                Scores
              </td>
            </tr>
            <tr className="bg-gray-100">
              <th></th>
              <td className="p-1">Label</td>
              {table?.columns.map((col) => (
                <td key={col} className="p-1 capitalize font-semibold">
                  {col}
                </td>
              ))}
              <td className="p-1 text-center">Recall</td>
              <td className="p-1 text-center">f1</td>
            </tr>
          </thead>
          <tbody>
            {table.data.map((row, rowIndex) => (
              <tr key={table.index[rowIndex]}>
                {rowIndex === 0 && (
                  <td rowSpan={colCount} className="bg-blue-100 font-semibold text-center p-3">
                    Truth
                  </td>
                )}
                <td className="font-medium p-1">{table.index[rowIndex]}</td>
                {row.map((cell, colIndex) => (
                  <td key={colIndex} className="p-1 text-center">
                    {colIndex === row.length - 1 ? cell : <b>{cell}</b>}
                  </td>
                ))}

                <td className="p-1 text-center">
                  {scores.recall_label && scores.recall_label[labels[rowIndex]]}
                </td>
                <td className="p-1 text-center">
                  {scores.f1_label && scores.f1_label[labels[rowIndex]]}
                </td>
              </tr>
            ))}
            <tr>
              <td></td>
              <td className="p-1 text-center">f1</td>
              {table.columns.map((col, colIndex) => (
                <td key={colIndex} className="p-1 text-center">
                  {scores.f1_label && scores.f1_label[col]}
                </td>
              ))}
            </tr>
            <tr>
              <td></td>
              <td className="p-1 text-center">Precision</td>
              {table.columns.map((col, colIndex) => (
                <td key={colIndex} className="p-1 text-center">
                  {scores.precision_label && scores.precision_label[col]}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
};
