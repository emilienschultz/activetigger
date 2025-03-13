import { FC, useEffect, useState } from 'react';
import DataGrid, { Column, RenderEditCellProps } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { useBlocker, useParams } from 'react-router-dom';

import Highlighter from 'react-highlight-words';
import { MdSkipNext, MdSkipPrevious } from 'react-icons/md';

import { Modal } from 'react-bootstrap';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { Link } from 'react-router-dom';
import { useAddTableAnnotations, useTableElements } from '../../core/api';
import { useAppContext } from '../../core/context';
import { AnnotationModel } from '../../types';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';
import { ProjectionManagement } from '../ProjectionManagement';
/**
 * Component to display the exploratory page
 */

interface Row {
  id: string;
  timestamp: string;
  labels: string;
  text: string;
}

export const ProjectExplorePage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, phase },
  } = useAppContext();

  // data modification management
  const [modifiedRows, setModifiedRows] = useState<Record<string, AnnotationModel>>({});

  const kindScheme =
    currentScheme && project
      ? (project.schemes.available[currentScheme]['kind'] as string) || 'multiclass'
      : 'multiclass';

  const blocker = useBlocker(({ currentLocation, nextLocation }) => {
    if (
      currentLocation.pathname !== nextLocation.pathname &&
      Object.values(modifiedRows).length > 0
    ) {
      return true;
    }
    return false;
  });

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme]['labels'] || [] : [];

  // selection elements
  const [page, setPage] = useState<number | null>(0);
  const [search, setSearch] = useState<string | null>(null);
  const [sample, setSample] = useState<string>('all');
  const [pageSize, setPageSize] = useState(20);

  // get API elements when table shape change
  const {
    table,
    getPage,
    total: totalElement,
  } = useTableElements(projectName, currentScheme, page, pageSize, search, sample, phase);

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
      key: 'id',
      name: 'id',
      resizable: true,
      width: 180,
      renderCell: (props) => (
        <div className={props.row.id in modifiedRows ? 'modified-cell' : ''}>
          <Link to={`/projects/${projectName}/annotate/${props.row.id}`}>{props.row.id}</Link>
        </div>
      ),
    },
    {
      key: 'labels',
      name: kindScheme === 'multiclass' ? 'Label ✎' : 'Label',
      resizable: true,
      renderCell: (props) => (
        <div style={{ textAlign: 'center', width: '100%' }}>{props.row.labels}</div>
      ),
      renderEditCell: kindScheme === 'multiclass' ? renderDropdown : undefined,
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
            userSelect: 'none',
          }}
        >
          <Highlighter
            highlightClassName="Search"
            searchWords={search && isValidRegex(search) ? [search.replace('ALL:', '')] : []}
            autoEscape={false}
            textToHighlight={props.row.text}
            highlightStyle={{
              backgroundColor: 'yellow',
              margin: '0px',
              padding: '0px',
            }}
            caseSensitive={true}
          />
        </div>
      ),
    },
    {
      key: 'comment',
      name: 'Comment',
      resizable: true,
      width: 100,
    },
    { key: 'timestamp', name: 'Changed', resizable: true, width: 100 },
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
            [row.id]: {
              element_id: row.id,
              label: event.target.value,
              scheme: currentScheme as string,
              project_slug: projectName as string,
              dataset: phase,
            },
          }));
        }}
        autoFocus
      >
        <option></option>
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
    phase || null,
  );
  function validateChanges() {
    addTableAnnotations(Object.values(modifiedRows)); // send the modifications
    setModifiedRows({}); // reset modified rows
  }

  function range(start: number, end: number) {
    return Array.from({ length: end - start }, (_, i) => start + i);
  }

  const isValidRegex = (pattern: string) => {
    try {
      new RegExp(pattern);
      return true;
    } catch (e) {
      return false;
    }
  };

  if (!projectName) return null;
  if (!project) return null;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explore">
      <div className="container-fluid">
        <Tabs id="explore" className="mt-3" defaultActiveKey="tabular">
          <Tab title="Tabular" eventKey="tabular">
            <div className="row mt-3">
              {phase == 'test' && (
                <div className="alert alert-warning">
                  Test mode activated - you are annotating the test set
                </div>
              )}
              <div className="col-12">
                {!isValidRegex(search || '') && (
                  <div className="alert alert-danger">Regex not valid</div>
                )}
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
                        value={pageSize}
                      >
                        {[10, 20, 50, 100].map((e) => (
                          <option key={e}>{e}</option>
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

                      <select
                        className="form-select w-25 mx-2"
                        onChange={(e) => setSample(e.target.value)}
                        value={sample}
                      >
                        {['tagged', 'untagged', 'all', 'recent'].map((e) => (
                          <option key={e}>{e}</option>
                        ))}
                      </select>
                      <input
                        className="form-control"
                        placeholder="Regex search to filter on text / for both text and label, use ALL: to start"
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
                <div className="d-flex justify-content-center mt-3 align-items-center">
                  <button
                    className="btn"
                    onClick={() => (page && page > 1 ? setPage(page - 1) : setPage(1))}
                  >
                    <MdSkipPrevious size={30} />
                  </button>{' '}
                  Change page{' '}
                  <button className="btn" onClick={() => (page ? setPage(page + 1) : setPage(1))}>
                    <MdSkipNext size={30} />
                  </button>
                </div>
              </div>
            </div>
            <Modal show={blocker.state === 'blocked'} onHide={blocker.reset}>
              <Modal.Header>
                <Modal.Title>You are leaving the page</Modal.Title>
              </Modal.Header>
              <Modal.Body>
                Do you want to change the page ? You will lose all your changes if you proceed
              </Modal.Body>
              <Modal.Footer>
                <button onClick={blocker.reset}>No</button>
                <button onClick={blocker.proceed}>Yes</button>
              </Modal.Footer>
            </Modal>
          </Tab>
          <Tab title="Visualization" eventKey="visualisation">
            <ProjectionManagement
              projectName={projectName || null}
              currentScheme={currentScheme || null}
              availableFeatures={project?.features.available ? project?.features.available : []}
            />
          </Tab>
        </Tabs>
      </div>
    </ProjectPageLayout>
  );
};
