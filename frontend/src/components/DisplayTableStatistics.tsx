import cx from 'classnames';
import { FC } from 'react';
import { MLStatisticsModel } from '../types';
import { useWindowSize } from 'usehooks-ts';

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
  const { width: widthWindow } = useWindowSize();
  const table = scores.table ? (scores.table as unknown as TableModel) : null;
  const labels = (scores?.table?.index as unknown as string[]) || [];
  const nLabels = Object.entries(labels).length;
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

  const isDiag = (colIndex: number, rowIndex: number) => {
    return colIndex < nLabels - 1 && colIndex === rowIndex;
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

  const limitLabelSize = (label: string) => {
    const maxLength = widthWindow > 1000 ? 8 : widthWindow > 500 ? 5 : 4;
    if (label) {
      if (label.length > maxLength) {
        return label.slice(0, maxLength - 3) + '..' + label.slice(-1);
      } else {
        return label;
      }
    } else {
      return 'Loading...';
    }
  };

  return (
    <>
      {table && (
        <div id={`DisplayTableStatisticsRENEW-${nLabels}`}>
          <div className="row">
            <div className="main row">
              <div id="truth-container" style={{ height: `${(nLabels + 2) * 30}px` }}>
                <span style={{ height: `${nLabels * 30}px` }}>Truth</span>
              </div>
              <div className="table">
                <div className="row">
                  {/* Predicted overlay row */}
                  <div className="table-cell label-column-left"></div>
                  <div
                    className="table-cell"
                    style={{ flex: `0 ${1 / nLabels} auto` }}
                    id="overlay-label"
                  >
                    Predicted
                  </div>
                </div>
                <div className="row">
                  {/* Labels row */}
                  <div className="table-cell label-column-left label-name"></div>
                  {labels.map((label) => (
                    <div className="table-cell label-name">{limitLabelSize(label)}</div>
                  ))}
                </div>
                {table.data.map((row, rowIndex) => (
                  // All data
                  <div className="row">
                    <div className="table-cell label-column-left label-name">
                      {limitLabelSize(table.index[rowIndex])}
                    </div>
                    {row.map((cell, colIndex) => (
                      <div
                        className={cx('table-cell', isDiag(colIndex, rowIndex) ? ' diag-cell' : '')}
                      >
                        {cell}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="score-left">
              <div className="table">
                <div className="row">
                  <div className="table-cell" style={{ flex: `0 .5 auto` }} id="overlay-label">
                    Scores
                  </div>
                  <div className="table-cell" style={{ flex: `0 1 auto` }}></div>
                </div>
                <div className="row">
                  <div className="table-cell">{limitLabelSize('Recall')}</div>
                  <div className="table-cell">{limitLabelSize('F1 Score')}</div>
                  <div className="table-cell"></div>
                </div>
                {table.data.map((row, rowIndex) => (
                  <div className="row">
                    <div className="table-cell">
                      {scores.recall_label && displayScore(scores.recall_label[labels[rowIndex]])}
                    </div>
                    <div className="table-cell">
                      {scores.f1_label && displayScore(scores.f1_label[labels[rowIndex]])}
                    </div>
                    <div className="table-cell label-name recall">
                      {table.index[rowIndex] !== 'Total'
                        ? limitLabelSize(table.index[rowIndex])
                        : ''}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="score-bottom row">
            <div id="truth-container" style={{ height: `${2 * 30}px` }}>
              <span style={{ height: `${2 * 30}px` }}>Scores</span>
            </div>
            <div className="table">
              <div className="row">
                <div className="table-cell label-column-left">{limitLabelSize('Accuracy')}</div>
                {table.columns.map((col, colIndex) => (
                  <div key={colIndex} className="table-cell">
                    {scores.f1_label && displayScore(scores.f1_label[col])}
                  </div>
                ))}
              </div>
              <div className="row">
                <div className="table-cell label-column-left">{limitLabelSize('F1-Score')}</div>
                {table.columns.map((col, colIndex) => (
                  <div key={colIndex} className="table-cell">
                    {scores.f1_label && displayScore(scores.f1_label[col])}
                  </div>
                ))}
              </div>
              <div className="row">
                <div className="table-cell label-column-left"></div>
                {table.columns.map((col, colIndex) => (
                  <div key={colIndex} className="table-cell label-name recall">
                    {col !== 'Total' ? limitLabelSize(col) : ''}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      <div id="DisplayTableStatistics" style={{ display: 'none' }}>
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
                {table?.columns.map((colName, colIndex) => (
                  <th
                    key={colName}
                    className={cx(
                      'text-center px-2',
                      colIndex === table?.columns.length - 1 ? '' : 'fw-bold',
                    )}
                  >
                    {limitLabelSize(colName)}
                  </th>
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
                  <td
                    className={cx(
                      'text-end px-2',
                      rowIndex === table.data.length - 1 ? '' : 'fw-bold',
                    )}
                  >
                    {limitLabelSize(table.index[rowIndex])}
                  </td>
                  {row.map((cell, colIndex) => (
                    <td
                      key={colIndex}
                      className={cx(
                        'cell',
                        isLabelColumn(colIndex, rowIndex, labels) ? ' label-col' : '',
                        isDiag(colIndex, rowIndex) ? ' diag-cell' : '',
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
                  <td className="name-recall">
                    {rowIndex === Object.entries(labels).length - 1
                      ? ''
                      : limitLabelSize(table.index[rowIndex])}
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
                <td className="text-end">Precision</td>
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
                <td className="text-end">F1</td>
                {table.columns.map((col, colIndex) => (
                  <td key={colIndex} className="cell">
                    {scores.f1_label && displayScore(scores.f1_label[col])}
                  </td>
                ))}
                <td></td>
                <td></td>
                <td></td>
              </tr>
              <tr>
                <td></td>
                <td></td>
                {table.columns.map((col, colIndex) => (
                  <td key={colIndex} className="name-recall">
                    {colIndex === Object.entries(labels).length - 1 ? '' : limitLabelSize(col)}
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
    </>
  );
};
