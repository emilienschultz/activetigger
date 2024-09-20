import { FC, useEffect, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import DataGrid, { Column } from 'react-data-grid';
import { useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import { useGenerate, useGeneratedElements, useGetGenerationsFile } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

// TODO
// interrupt button using event
// - better table
// - how to merge the results in the database
// --- button to validate the automatic mergin + rule to force the labels in the scheme

interface Row {
  time: string;
  index: string;
  prompt: string;
  answer: string;
  endpoint: string;
}

export const GenPage: FC = () => {
  const { projectName } = useParams();
  const { authenticatedUser } = useAuth();
  const {
    appContext: { generateConfig, currentScheme, currentProject },
    setAppContext,
  } = useAppContext();

  // currently generating for the user
  const isGenerating =
    authenticatedUser?.username &&
    currentProject?.generations.training &&
    Object.keys(currentProject?.generations.training).includes(authenticatedUser?.username);

  // call api to post generation
  const { generate } = useGenerate(
    projectName || null,
    currentScheme || null,
    generateConfig.api || null,
    generateConfig.endpoint || null,
    generateConfig.n_batch || null,
    generateConfig.prompt || null,
    generateConfig.selection_mode || null,
    generateConfig.token,
  );

  // call api to get a sample of elements
  const { generated } = useGeneratedElements(projectName || null, 10, isGenerating || false);

  // call api to download a batch of elements
  const { getGenerationsFile } = useGetGenerationsFile(projectName || null);
  const [numberElements, setNumberElements] = useState<number>(10);

  useEffect(() => {
    if (!generateConfig.api)
      setAppContext((prev) => ({
        ...prev,
        generateConfig: { ...generateConfig, api: 'ollama' },
      }));
  }, [generateConfig, setAppContext]);

  console.log(generateConfig);

  const columns: readonly Column<Row>[] = [
    {
      name: 'Time',
      key: 'time',
      resizable: true,
    },
    {
      name: 'Id',
      key: 'index',
      resizable: true,
    },
    {
      name: 'Answer',
      key: 'answer',
      resizable: true,
    },
    {
      name: 'Prompt',
      key: 'prompt',
      resizable: true,
    },

    {
      name: 'Endpoint',
      key: 'endpoint',
      resizable: true,
    },
  ];

  console.log(currentProject);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="generate">
      <div className="container-fluid mt-3">
        <div className="row">
          <div className="alert alert-warning" role="alert">
            This page is under developement
          </div>{' '}
          <Tabs id="panel" className="mb-1" defaultActiveKey="api">
            <Tab eventKey="api" title="API config">
              {' '}
              <div className="form-floating mt-3">
                <select className="form-control" id="api">
                  <option key="ollama">Ollama</option>
                </select>
                <label htmlFor="api">API </label>
              </div>
              <div className="form-floating mt-3">
                <input
                  type="text"
                  id="endpoint"
                  className="form-control  mt-3"
                  placeholder="enter the url of the endpoint"
                  value={generateConfig.endpoint || undefined}
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      generateConfig: { ...generateConfig, endpoint: e.target.value },
                    }));
                  }}
                />
                <label htmlFor="endpoint">Endpoint </label>
              </div>
            </Tab>
            <Tab eventKey="selection" title="Selection mode">
              Current scheme : {currentScheme}
              <div className="form-floating mt-3">
                <select
                  id="mode"
                  className="form-control mt-3"
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      generateConfig: { ...generateConfig, selection_mode: e.target.value },
                    }));
                  }}
                >
                  <option key="all">all</option>
                  <option key="untagged">untagged</option>
                </select>
                <label htmlFor="mode">Sample </label>
              </div>
              <div className="form-floating mt-3">
                <input
                  type="number"
                  id="batch"
                  className="form-control mt-3"
                  value={generateConfig.n_batch}
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      generateConfig: { ...generateConfig, n_batch: Number(e.target.value) },
                    }));
                  }}
                />
                <label htmlFor="batch">N elements to annotate </label>
              </div>
            </Tab>
          </Tabs>
          <div className="explanations mt-5">
            Craft your prompt with the element #INSERTTEXT to insert text
          </div>
          <div className="form-floating mt-2">
            <textarea
              id="prompt"
              rows={5}
              placeholder="Enter your prompt"
              className="form-control"
              style={{ height: '200px' }}
              value={generateConfig.prompt || undefined}
              onChange={(e) => {
                setAppContext((prev) => ({
                  ...prev,
                  generateConfig: { ...generateConfig, prompt: e.target.value },
                }));
              }}
            />
            <label htmlFor="prompt">Prompt </label>
          </div>
          <div className="col-12 text-center">
            {!!isGenerating && <PulseLoader />}
            <button className="btn btn-primary  mt-3" onClick={generate} disabled={!!isGenerating}>
              Generate
            </button>
          </div>
        </div>
        <hr />
        <div className="col-12 d-flex align-items-center justify-content-center">
          <span>Number of last generated elements to download</span>
          <input
            type="number"
            placeholder="Number of last generated elements to download"
            className="form-control m-4"
            style={{ width: '100px' }}
            value={numberElements}
            onChange={(e) => setNumberElements(Number(e.target.value))}
          />
          <button className="btn btn-secondary" onClick={() => getGenerationsFile(numberElements)}>
            Download
          </button>
        </div>
        <div className="explanations">Last generated content for the current user</div>
        <DataGrid
          className="fill-grid"
          columns={columns}
          rows={(generated as unknown as Row[]) || []}
        />
      </div>
    </ProjectPageLayout>
  );
};
