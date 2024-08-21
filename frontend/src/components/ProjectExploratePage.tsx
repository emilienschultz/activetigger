import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { FC, useEffect, useState } from 'react';
import DataGrid, { Column, RenderEditCellProps } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { useParams } from 'react-router-dom';

import { useAddTableAnnotations, useTableElements } from '../core/api';
import { useAppContext } from '../core/context';
import { AnnotationModel } from '../types';
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
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  const [page, setPage] = useState<number | null>(0);
  const [pageSize, setPageSize] = useState(20);

  // data modification management
  const [modifiedRows, setModifiedRows] = useState<Record<string, AnnotationModel>>({});

  // get API elements when table shape change
  const {
    table,
    getPage,
    total: totalElement,
  } = useTableElements(projectName, currentScheme, page, pageSize);

  const [rows, setRows] = useState<Row[]>([]);

  // update rows only when a even trigger the update table
  useEffect(() => {
    if (table) {
      setRows(table as Row[]);
    }
  }, [table]);

  useEffect(() => {
    if (page !== null) getPage({ pageIndex: page, pageSize });
  }, [page, pageSize, getPage]);

  if (!projectName) return null;
  if (!currentScheme) return null;
  if (!project) return null;

  // define table
  const columns: readonly Column<Row>[] = [
    {
      key: 'index',
      name: 'ID',
      resizable: true,
      width: 180,
      renderCell: (props) =>
        props.row.index in modifiedRows ? (
          <div className="modified-cell">{props.row.index}</div>
        ) : (
          <div>{props.row.index}</div>
        ),
    },
    { key: 'timestamp', name: 'Timestamp', resizable: true, width: 100 },
    { key: 'labels', name: 'Label', resizable: true, renderEditCell: renderDropdown, width: 100 },
    {
      key: 'text',
      name: 'Text',
      resizable: true,

      renderCell: (props) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
          }}
        >
          {props.row.text}
        </div>
      ),
    },
  ];

  // specific function to have a select component
  function renderDropdown({ row, onRowChange }: RenderEditCellProps<Row>) {
    return (
      <select
        value={row.labels}
        onChange={(event) => {
          onRowChange({ ...row, labels: event.target.value }, true);
          setModifiedRows((prevState) => ({
            ...prevState,
            [row.index]: {
              element_id: row.index,
              label: event.target.value,
              scheme: currentScheme as string,
              project_slug: projectName as string,
            },
          }));
          console.log(modifiedRows);
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

  // send changes
  const { addTableAnnotations } = useAddTableAnnotations(projectName, currentScheme);
  function validateChanges() {
    addTableAnnotations(Object.values(modifiedRows)); // send the modifications
    setModifiedRows({}); // reset modified rows
  }

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explorate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <h2 className="subsection">Data exploration</h2>

            {table && (
              <div>
                <div className="d-flex align-items-center justify-content-between mb-3">
                  {Object.keys(modifiedRows).length > 0 && (
                    <button onClick={validateChanges}>Validate changes</button>
                  )}
                  <span>Total elements : {totalElement}</span>
                  <span>Page size</span>
                  <select
                    onChange={(e) => {
                      setPage(1);
                      setPageSize(Number(e.target.value));
                    }}
                  >
                    {[10, 20, 50, 100].map((e) => (
                      <option key={e} selected={e === pageSize}>
                        {e}
                      </option>
                    ))}
                  </select>
                  <label>Page</label>
                  <input
                    // todo change this input to a select plus previous/next button
                    className="from-control"
                    type="number"
                    step="1"
                    min="1"
                    max={totalElement > 0 ? Math.ceil(totalElement / pageSize) : 1}
                    value={page || ''}
                    onChange={(e) => {
                      if (e.target.value === '') setPage(null);
                      let val = Number(e.target.value);
                      // user can input float number 5.6 in which case we keep 5
                      val = Math.floor(val);
                      if (val > 0 && val <= Math.ceil(totalElement / pageSize)) setPage(val);
                    }}
                  />
                </div>

                <DataGrid
                  className="fill-grid"
                  columns={columns}
                  rows={rows}
                  rowHeight={80}
                  onRowsChange={(e) => {
                    setRows(e);
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
