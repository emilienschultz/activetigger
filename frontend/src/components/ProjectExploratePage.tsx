import { ColDef, ColGroupDef } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { AgGridReact } from 'ag-grid-react';
import { FC, useState } from 'react';
import DataGrid, { RenderEditCellProps, SelectColumn, textEditor } from 'react-data-grid';
import type { Row } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { useParams } from 'react-router-dom';

import { useGetTableElements } from '../core/api';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the features page
 */

const labels = ['Dr.', 'Mr.', 'Mrs.', 'Miss', 'Ms.'] as const;

export function renderDropdown({ row, onRowChange }: RenderEditCellProps<Row>) {
  return (
    <select
      value={row.labels}
      onChange={(event) => onRowChange({ ...row, labels: event.target.value }, true)}
      autoFocus
    >
      {labels.map((l) => (
        <option key={l} value={l}>
          {l}
        </option>
      ))}
    </select>
  );
}

export const ProjectExploratePage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;
  const {
    appContext: { currentScheme },
    setAppContext,
  } = useAppContext();
  if (!currentScheme) return null;
  const [rowData, setRowData] = useState([]);
  const [page, setPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [pageSize, setPageSize] = useState(20);

  const [loading, setLoading] = useState(false);
  const { table } = useGetTableElements(
    projectName,
    currentScheme,
    page * pageSize,
    (page + 1) * pageSize,
  );
  console.log(table);
  console.log(pageSize);
  console.log(page);

  const columns = [
    { key: 'index', name: 'ID', resizable: true },
    { key: 'timestamp', name: 'Timestamp', resizable: true },
    { key: 'labels', name: 'Label', resizable: true, renderEditCell: renderDropdown },
    { key: 'text', name: 'Text', resizable: true },
  ];

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explorate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-0"></div>
          <div className="col-10">
            <h2 className="subsection">Data exploration</h2>

            {table && (
              <div>
                <div className="d-flex align-items-center justify-content-between mb-3">
                  <span>Total elements : {table['total']}</span>
                  <span>Page size</span>
                  <select onChange={(e) => setPageSize(Number(e.target.value))}>
                    {[10, 20, 50, 100].map((e) => (
                      <option key={e}>{e}</option>
                    ))}
                  </select>
                  <label>Page</label>
                  <input
                    type="number"
                    step="1"
                    value={page}
                    onChange={(e) => {
                      let val = Number(e.target.value);
                      if (val < table['total'] / pageSize) setPage(val);
                    }}
                  />
                </div>
                <div>
                  <DataGrid columns={columns} rows={table['items'] ? table['items'] : []} />
                </div>
              </div>
            )}
          </div>{' '}
        </div>
      </div>
    </ProjectPageLayout>
  );
};
