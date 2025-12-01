import cx from 'classnames';
import { FC } from 'react';
import { useWindowSize } from 'usehooks-ts';
import { MLStatisticsModel } from '../types';
import React from 'react';

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
        <div id={`DisplayTableStatistics-${nLabels}`}>
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
                    <div className="table-cell label-name" key={label}>
                      {limitLabelSize(label)}
                    </div>
                  ))}
                </div>
                {table.data.map((row, rowIndex) => (
                  // All data
                  <div className="row" key={rowIndex}>
                    <div className="table-cell label-column-left label-name">
                      {limitLabelSize(table.index[rowIndex])}
                    </div>
                    {row.map((cell, colIndex) => (
                      <div
                        className={cx('table-cell', isDiag(colIndex, rowIndex) ? ' diag-cell' : '')}
                        key={colIndex}
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
                {table.data.map((_, rowIndex) => (
                  <div className="row" key={rowIndex}>
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
    </>
  );
};
