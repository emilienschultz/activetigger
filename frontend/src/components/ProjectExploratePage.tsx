import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { FC, useEffect, useState } from 'react';
import DataGrid, { Column, RenderEditCellProps, SelectColumn, textEditor } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { useParams } from 'react-router-dom';

import { useGetTableElements } from '../core/api';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the exploratory page
 */

interface Row {
  index: string;
  timestamp: string;
  labels: string;
  text: string;
}

export const ProjectExploratePage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();
  if (!currentScheme) return null;
  if (!project) return null;

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(20);

  // get API elements when table shape change
  const { table } = useGetTableElements(
    projectName,
    currentScheme,
    page * pageSize,
    (page + 1) * pageSize,
  );

  const [rows, setRows] = useState(table ? table['items'] : []);

  // update rows only when a even trigger the update table
  useEffect(() => {
    if (table) {
      setRows(table['items']);
      console.log('updage table');
    }
  }, [table]);

  // define table
  const columns: readonly Column<Row>[] = [
    { key: 'index', name: 'ID', resizable: true },
    { key: 'timestamp', name: 'Timestamp', resizable: true },
    { key: 'labels', name: 'Label', resizable: true, renderEditCell: renderDropdown },
    { key: 'text', name: 'Text', resizable: true },
  ];

  // specific function to have a select component
  function renderDropdown({ row, onRowChange }: RenderEditCellProps<Row>) {
    return (
      <select
        value={row.labels}
        onChange={(event) => {
          onRowChange({ ...row, labels: event.target.value }, true);
          console.log(event.target.value);
        }}
        autoFocus
      >
        {availableLabels.map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>
    );
  }

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
                <div className="rdg rnvodz5 fill-grid">
                  <DataGrid
                    columns={columns}
                    rows={rows}
                    onRowsChange={(e) => {
                      setRows(e);
                      console.log(e);
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
