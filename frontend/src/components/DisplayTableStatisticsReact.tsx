import cx from 'classnames';
import { FC, useMemo } from 'react';
import DataTable from 'react-data-table-component';
import { useWindowSize } from 'usehooks-ts';
import { useAppContext } from '../core/context';
import { reorderLabels } from '../core/utils';
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

type rowData = Record<string, string | number>;

interface confusionMatrixTableType {
  headers: string[];
  data: rowData;
}

interface scoresTableType {
  headers: string[];
  data: rowData;
}

export const DisplayTableStatisticsReact: FC<DisplayTableStatisticsProps> = ({ scores }) => {
  const { width: widthWindow } = useWindowSize();
  const {
    appContext: { displayConfig },
  } = useAppContext();
  const table = scores.table ? (scores.table as unknown as TableModel) : null;

  // sort labels and build a permutation to reorder data rows/columns accordingly
  const reorderedLabels = useMemo<string[]>(
    () =>
      reorderLabels(
        (scores?.table?.index as unknown as string[]) || [],
        displayConfig.labelsOrder || [],
      ),
    [displayConfig.labelsOrder, scores],
  );
  const nLabels = Object.entries(reorderedLabels).length;

  // permutation: for each position in `labels`, the original index in table.index
  const perm = useMemo<number[]>(() => {
    if (!table) return [];
    return reorderedLabels.map((label) => table.index.indexOf(label));
  }, [reorderedLabels, table]);

  // reordered data: rows and columns permuted to match `labels` order
  const reorderedConfusionMatrix = useMemo<number[][]>(() => {
    if (!table) return [];
    return perm.map((origRow) => perm.map((origCol) => table.data[origRow][origCol]));
  }, [table, perm]);

  console.log(reorderedLabels);
  console.log('scores', scores);
  console.log(reorderedConfusionMatrix);

  const tableData: rowData[] = reorderedLabels.map((rowLabel, iRow) => {
    return {
      _rowLabel: rowLabel,
      ...Object.fromEntries(
        reorderedLabels.map((columnLabel, iCol) => {
          return [columnLabel, reorderedConfusionMatrix[iRow][iCol]];
        }),
      ),
    } as rowData;
  });

  const scoreColumns: string[] = ['Recall', 'Precision', 'F1'];
  const scoreData: rowData[] = reorderedLabels
    .filter((label) => label !== 'Total')
    .map((rowLabel, iRow) => {
      return {
        _rowLabel: rowLabel,
        Recall: scores.recall_label ? scores.recall_label[rowLabel] : 'NaN',
        Precision: scores.precision_label ? scores.precision_label[rowLabel] : 'NaN',
        F1: scores.f1_label ? scores.f1_label[rowLabel] : 'NaN',
      } as rowData;
    });

  console.log('scoreData', scoreData);

  return (
    <>
      {table && (
        <>
          <p>Working on it </p>
          <DataTable<Record<confusionMatrixTableType['headers'][number], string | number>>
            title="Confusion Matrix"
            columns={[
              { name: '', selector: (row) => row['_rowLabel'] },
              ...reorderedLabels.map((label) => {
                return {
                  name: label,
                  selector: (row: Record<string, string | number>) => row[label],
                };
              }),
            ]}
            data={tableData}
          />
          <DataTable<Record<scoresTableType['headers'][number], string | number>>
            title="Scores"
            columns={[
              { name: '', selector: (row) => row['_rowLabel'] },
              ...scoreColumns.map((score) => {
                return {
                  name: score,
                  selector: (row: Record<string, string | number>) => row[score],
                };
              }),
            ]}
            data={scoreData}
          />
        </>
      )}
    </>
  );
};
