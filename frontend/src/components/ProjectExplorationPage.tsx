import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { FC, useEffect, useState } from 'react';
import DataGrid, { Column, RenderEditCellProps } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { useParams } from 'react-router-dom';

import Highlighter from 'react-highlight-words';
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

export const ProjectExplorationPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  const [page, setPage] = useState<number | null>(0);
  const [search, setSearch] = useState<string | null>(null);
  const [sample, setSample] = useState<string>('all');
  const [pageSize, setPageSize] = useState(20);

  console.log(sample);

  // data modification management
  const [modifiedRows, setModifiedRows] = useState<Record<string, AnnotationModel>>({});

  // get API elements when table shape change
  const {
    table,
    getPage,
    total: totalElement,
  } = useTableElements(projectName, currentScheme, page, pageSize, search, sample);

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
    {
      key: 'labels',
      name: 'Label ✎',
      resizable: true,
      renderEditCell: renderDropdown,
      width: 100,
    },
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
          <Highlighter
            highlightClassName="Search"
            searchWords={search ? [search] : []}
            autoEscape={true}
            textToHighlight={props.row.text}
          />
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
  const { addTableAnnotations } = useAddTableAnnotations(
    projectName || null,
    currentScheme || null,
  );
  function validateChanges() {
    addTableAnnotations(Object.values(modifiedRows)); // send the modifications
    setModifiedRows({}); // reset modified rows
  }

  function range(start: number, end: number) {
    return Array.from({ length: end - start }, (_, i) => start + i);
  }

  if (!projectName) return null;
  if (!project) return null;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explorate">
      <div className="container-fluid">
        <div className="row mt-3">
          <div className="col-12">
            {currentScheme && table && (
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
                    className="form-select w-25"
                  >
                    {[10, 20, 50, 100].map((e) => (
                      <option key={e} selected={e === pageSize}>
                        {e}
                      </option>
                    ))}
                  </select>
                  <label>Page</label>
                  <select
                    className="form-select w-25"
                    onChange={(e) => {
                      setPage(Number(e.target.value));
                    }}
                    value={page || '1'}
                  >
                    {range(1, totalElement > 0 ? Math.ceil(totalElement / pageSize) : 1).map(
                      (v) => (
                        <option key={v}>{v}</option>
                      ),
                    )}
                  </select>
                </div>
                <div className="d-flex align-items-center justify-content-between mb-3">
                  <label>Filter</label>

                  <select className="form-control w-25" onChange={(e) => setSample(e.target.value)}>
                    {['tagged', 'untagged', 'all', 'recent'].map((e) => (
                      <option key={e}>{e}</option>
                    ))}
                  </select>
                  <input
                    className="form-control"
                    placeholder="You can use a regex search to filter"
                    onChange={(e) => setSearch(e.target.value)}
                  ></input>
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
