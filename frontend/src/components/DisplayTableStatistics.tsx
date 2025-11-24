import cx from 'classnames';
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
  const labels = (scores?.table?.index as unknown as string[]) || [];
  const colCount = table?.columns.length || 0;

  const isLabelColumn = (colIndex: number, rowIndex: number, labels: string[]) => {
    return (
      colIndex < Object.entries(labels).length - 1 && rowIndex < Object.entries(labels).length - 1
    );
  };

  const isTotalCell = (colIndex: number, rowIndex: number, labels: string[]) => {
    return (
      colIndex === Object.entries(labels).length - 1 ||
      rowIndex === Object.entries(labels).length - 1
    );
  };

  const isDiag = (colIndex: number, rowIndex: number, labels: string[]) => {
    return colIndex < Object.entries(labels).length - 1 && colIndex === rowIndex;
  };

  const displayScore = (score: number | undefined) => {
    if (typeof score === 'number') {
      if (score === 0) {
        return '0.00';
      } else if (score === 1) {
        return '1.00';
      } else {
        return (String(score) + '0').slice(0, 4);
      }
    }
    return '';
  };

  return (
    <div id="DisplayTableStatistics">
      {table && (
        <table>
          {title && (
            <caption className="caption-top text-lg font-medium mb-2 text-gray-700">
              {title}
            </caption>
          )}
          <thead>
            <tr>
              <td></td>
              <td></td>
              <td
                colSpan={Object.entries(labels).length}
                className="text-center p-2 border-bottom border-secondary"
              >
                Predicted
              </td>
              <td style={{ width: '20px' }}></td>
              <td colSpan={2} className="text-center p-2 border-bottom border-secondary">
                Scores
              </td>
            </tr>
            <tr className="bg-gray-100">
              <th></th>
              <th></th>
              {table?.columns.map((col, colIndex) => (
                <td
                  key={col}
                  className={cx(
                    'text-center ',
                    colIndex === table?.columns.length - 1 ? '' : 'fw-bold',
                  )}
                >
                  {col}
                </td>
              ))}
              <th></th>
              <th className="text-center fw-normal">Recall</th>
              <th className="text-center fw-normal">F1</th>
            </tr>
          </thead>
          <tbody>
            {table.data.map((row, rowIndex) => (
              <tr key={table.index[rowIndex]}>
                {rowIndex === 0 && (
                  <td rowSpan={colCount} className="rowspan-cell">
                    Truth
                  </td>
                )}
                <td className="font-medium p-1">
                  {rowIndex === table.data.length - 1 ? (
                    table.index[rowIndex]
                  ) : (
                    <b>{table.index[rowIndex]}</b>
                  )}
                </td>
                {row.map((cell, colIndex) => (
                  <td
                    key={colIndex}
                    className={cx(
                      'cell',
                      isLabelColumn(colIndex, rowIndex, labels) ? ' label-col' : '',
                      isDiag(colIndex, rowIndex, labels) ? ' diag-cell' : '',
                      isTotalCell(colIndex, rowIndex, labels) ? ' total-cell' : '',
                    )}
                  >
                    {cell}
                  </td>
                ))}
                <td></td>

                <td className="cell">
                  {scores.recall_label && displayScore(scores.recall_label[labels[rowIndex]])}
                </td>
                <td className="cell">
                  {scores.f1_label && displayScore(scores.f1_label[labels[rowIndex]])}
                </td>
              </tr>
            ))}

            <tr style={{ height: '20px' }}>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>

            <tr>
              <td rowSpan={2} className="rowspan-cell">
                {' '}
                Scores
              </td>
              <td>Precision</td>
              {table.columns.map((col, colIndex) => (
                <td key={colIndex} className="cell">
                  {scores.precision_label && displayScore(scores.precision_label[col])}
                </td>
              ))}
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>F1</td>
              {table.columns.map((col, colIndex) => (
                <td key={colIndex} className="cell">
                  {scores.f1_label && displayScore(scores.f1_label[col])}
                </td>
              ))}
              <td></td>
              <td></td>
              <td></td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
};
