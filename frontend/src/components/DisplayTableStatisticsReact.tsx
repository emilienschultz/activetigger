import { FC, useMemo, ReactNode } from 'react';
import DataTable, { TableColumn } from 'react-data-table-component';
import { useAppContext } from '../core/context';
import { reorderLabels } from '../core/utils';
import { MLStatisticsModel } from '../types';
import { CSSObject } from 'styled-components';

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

interface tableColumnsType {
  name: string;
  selector: (row: Record<string, string | number>) => number | string;
  minWidth?: string;
  center?: boolean;
  style?: CSSObject;
  grow?: number;
  cell?: (
    row: Record<string, string | number>,
    rowIndex: number,
    column: TableColumn<Record<string, string | number>>,
    id: string | number,
  ) => ReactNode;
}

export const DisplayTableStatisticsReact: FC<DisplayTableStatisticsProps> = ({ scores }) => {
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

  const cellFormat = (
    row: Record<string, string | number>,
    rowIndex: number,
    column: TableColumn<Record<string, string | number>>, // equivalent to tableColumnsType
    id: string | number,
  ) => {
    const columnName: string = column.name as unknown as string;
    const content = row[columnName];

    // Display labels vertical
    if (columnName.length === 0) return <>{row['_rowLabel']}</>;

    // Display total values in grey
    if (columnName === 'Total' || row['_rowLabel'] === 'Total')
      return <p style={{ margin: 'auto', color: '#909090' }}>{content}</p>;

    // Display diagonal values in green
    if (row['_rowLabel'] == columnName)
      return (
        <p
          style={{
            backgroundColor: '#90909040',
            margin: 'auto',
            padding: '.5rem 1rem',
            borderRadius: '.8rem',
            color: 'green',
            fontWeight: 'bold',
          }}
        >
          {content}
        </p>
      );
    // Should not happend
    return <p style={{ margin: 'auto', color: 'red', fontWeight: 'bold' }}>{content}</p>;
  };
  const [confusionMatrixColumns, confusionMatrixData] = useMemo<
    [tableColumnsType[], rowData[]]
  >(() => {
    const tableColumns: tableColumnsType[] = [
      {
        name: '',
        selector: (row: Record<string, string | number>) => row['_rowLabel'],
        minWidth: '8 rem',
        style: {
          fontWeight: 'bold',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        },
        grow: 2,
        cell: cellFormat,
      },
      ...reorderedLabels.map((label) => {
        return {
          name: label,
          selector: (row: Record<string, string | number>) => row[label],
          minWidth: 'fit-content',
          center: true,
          cell: cellFormat,
        };
      }),
    ];
    const tableData = reorderedLabels.map((rowLabel, iRow) => {
      return {
        _rowLabel: rowLabel,
        ...Object.fromEntries(
          reorderedLabels.map((columnLabel, iCol) => {
            return [columnLabel, reorderedConfusionMatrix[iRow][iCol]];
          }),
        ),
      } as rowData;
    });
    return [tableColumns, tableData];
  }, [reorderedLabels, reorderedConfusionMatrix]);

  const [scoresColumns, scoresData] = useMemo<[tableColumnsType[], rowData[]]>(() => {
    const tableColumns: tableColumnsType[] = [
      {
        name: '',
        selector: (row) => row['_rowLabel'],
        minWidth: '8 rem',
        style: {
          fontWeight: 'bold',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        },
        grow: 2,
      },
      ...['Recall', 'Precision', 'F1'].map((score) => {
        return {
          name: score,
          selector: (row: Record<string, string | number>) => row[score],
          minWidth: 'fit-content',
          center: true,
        };
      }),
    ];
    const scoreData: rowData[] = reorderedLabels
      .filter((label) => label !== 'Total')
      .map((rowLabel) => {
        return {
          _rowLabel: rowLabel,
          Recall: scores.recall_label ? scores.recall_label[rowLabel] : 'NaN',
          Precision: scores.precision_label ? scores.precision_label[rowLabel] : 'NaN',
          F1: scores.f1_label ? scores.f1_label[rowLabel] : 'NaN',
        } as rowData;
      });
    return [tableColumns, scoreData];
  }, [reorderedLabels, scores]);

  return (
    <>
      {table && (
        <>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'center',
              alignItems: 'start',
            }}
          >
            <div style={{ flex: '2 1 50px', margin: '1rem 2rem' }}>
              <DataTable<Record<confusionMatrixTableType['headers'][number], string | number>>
                columns={confusionMatrixColumns}
                data={confusionMatrixData}
              />
            </div>
            <div style={{ flex: '1 2 50px', margin: '1rem 2rem' }}>
              <DataTable<Record<scoresTableType['headers'][number], string | number>>
                columns={scoresColumns}
                data={scoresData}
              />
            </div>
          </div>
        </>
      )}
    </>
  );
};
