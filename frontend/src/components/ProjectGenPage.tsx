import { FC, useEffect, useState } from 'react';
import DataGrid, { Column } from 'react-data-grid';
import { useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import {
  useGenerate,
  useGeneratedElements,
  useGetGenerationsFile,
  useGetGenModels,
  useStopGenerate,
} from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';
import { GenModel } from '../types';
import { IoIosAddCircle } from 'react-icons/io';
import { GenModelSetupForm } from './forms/GenModelSetupForm';

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

const AddButton: FC = ({ showAddForm }) => (
  <button className="btn btn-primary" onClick={showAddForm}>
    <IoIosAddCircle className="m-1" size={30} />
    Add a generative model
  </button>
);

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

  const { stopGenerate } = useStopGenerate(projectName || null);

  // call api to get a sample of elements
  const { generated } = useGeneratedElements(projectName || null, 10, isGenerating || false);

  // GenModels
  const { models } = useGetGenModels();
  const [configuredModels, setConfigureModels] = useState<GenModels[]>([]);
  const [showForm, setShowForm] = useState<boolean>(false);

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

  const showAddForm = () => {
    setShowForm(true);
  };

  const hideForm = () => {
    setShowForm(false);
  };

  const addModel = (model: GenModel) => {
    setConfigureModels([...configuredModels, model]);
    setShowForm(false);
  };

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="generate">
      <div className="container-fluid mt-3">
        <div className="row">
          <div className="alert alert-warning" role="alert">
            This page is under developement
          </div>
          <div className="row"> Current scheme : {currentScheme}</div>
        </div>

        {configuredModels.length === 0 ? (
          <>
            {showForm ? (
              <GenModelSetupForm add={addModel} cancel={hideForm} />
            ) : (
              <>
                <p>No generative models assigned to this project</p>
                <AddButton showAddForm={showAddForm}></AddButton>
              </>
            )}
          </>
        ) : (
          <>
            <div className="row row-gap-2">
              <div className="col-6">
                <div className="form-floating">
                  <select id="model" className="form-select">
                    {configuredModels.map((model) => (
                      <option key="{model.id}">{model.name}</option>
                    ))}
                  </select>
                  <label htmlFor="model">Model</label>
                </div>
              </div>
              <div className="col-6">
                <div className="form-floating">
                  <select
                    id="mode"
                    className="form-select"
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
              </div>
              <div className="col-6">
                <AddButton showAddForm={showAddForm}></AddButton>
              </div>
              <div className="col-6">
                <div className="form-floating">
                  <input
                    type="number"
                    id="batch"
                    className="form-control"
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
              </div>
              <hr className="mt-3" />
              <div className="explanations mt-3">
                Craft your prompt with the element #INSERTTEXT to insert text
              </div>
              <div className="form-floating mt-2">
                <textarea
                  id="prompt"
                  rows={5}
                  placeholder="Enter your prompt"
                  className="form-control"
                  style={{ height: '200px' }}
                  value={generateConfig.prompt || ''}
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
                {isGenerating ? (
                  <div>
                    <PulseLoader />
                    <button className="btn btn-primary mt-3" onClick={stopGenerate}>
                      Stop
                    </button>
                  </div>
                ) : (
                  <button
                    className="btn btn-primary mt-3"
                    onClick={generate}
                    disabled={!!isGenerating}
                  >
                    Generate
                  </button>
                )}
                <div className="explanations"> It can take some time if you have a large batch</div>
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
                value={numberElements || 10}
                onChange={(e) => setNumberElements(Number(e.target.value))}
              />
              <button
                className="btn btn-secondary"
                onClick={() => getGenerationsFile(numberElements)}
              >
                Download
              </button>
            </div>
            <div className="explanations">Last generated content for the current user</div>
            <DataGrid
              className="fill-grid"
              columns={columns}
              rows={(generated as unknown as Row[]) || []}
            />
          </>
        )}
      </div>
    </ProjectPageLayout>
  );
};
