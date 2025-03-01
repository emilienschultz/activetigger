import { ChangeEvent, FC, useEffect, useState } from 'react';
import DataGrid, { Column } from 'react-data-grid';
import { BsSave2 } from 'react-icons/bs';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineSparkles } from 'react-icons/hi';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';
import {
  createGenModel,
  deleteGenModel,
  getProjectGenModels,
  useDeletePrompts,
  useGenerate,
  useGeneratedElements,
  useGetGenerationsFile,
  useGetPrompts,
  useSavePrompts,
  useStopGenerate,
} from '../../core/api';
import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';
import { GenModel, SupportedAPI } from '../../types';
import { GenModelSetupForm } from './../forms/GenModelSetupForm';
import { ProjectPageLayout } from './../layout/ProjectPageLayout';

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

const AddButton: FC<{ showAddForm: () => void }> = ({ showAddForm }) => (
  <button onClick={showAddForm} className="btn btn-primary">
    <FaPlusCircle size={20} /> Add model
  </button>
);

export const GenPage: FC = () => {
  const { projectName } = useParams() as { projectName: string };
  const { authenticatedUser } = useAuth();
  const {
    appContext: { generateConfig, currentScheme, currentProject },
    setAppContext,
  } = useAppContext();

  // GenModels
  const [configuredModels, setConfigureModels] = useState<Array<GenModel & { api: string }>>([]);
  const [selectedModel, setSelectedModel] = useState<number>();
  const [showForm, setShowForm] = useState<boolean>(false);

  // currently generating for the user
  const [isGenerating, setGenerating] = useState<boolean>(false);
  useEffect(() => {
    const trainings: Array<{ user: string }> = currentProject?.generations.training || [];
    setGenerating(
      authenticatedUser?.username !== undefined &&
        trainings.length !== 0 &&
        trainings.some((training) => training.user === authenticatedUser?.username),
    );
  }, [authenticatedUser, currentProject]);

  // call api to post generation
  const { generate } = useGenerate(
    projectName || null,
    currentScheme || null,
    selectedModel || null,
    generateConfig.n_batch || null,
    generateConfig.prompt || null,
    generateConfig.selection_mode || null,
    generateConfig.token,
  );

  const { stopGenerate } = useStopGenerate(projectName || null);

  // call api to get a sample of elements
  const { generated } = useGeneratedElements(projectName || null, 10, isGenerating);

  // call api to download a batch of elements
  const { getGenerationsFile } = useGetGenerationsFile(projectName || null);
  const [numberElements, setNumberElements] = useState<number>(10);

  // call api for prompts
  const { prompts, reFetchPrompts } = useGetPrompts(projectName || null);
  const savePrompts = useSavePrompts(projectName || null);
  const deletePrompts = useDeletePrompts(projectName || null);

  useEffect(() => {
    if (!generateConfig.api)
      setAppContext((prev) => ({
        ...prev,
        generateConfig: { ...generateConfig, api: 'ollama' },
      }));
  }, [generateConfig, setAppContext]);

  useEffect(() => {
    const fetchModels = async () => {
      const models = await getProjectGenModels(projectName);
      setConfigureModels(models);
      if (models.length > 0) setSelectedModel(models[0].id);
    };
    fetchModels();
  }, [projectName]);

  const columns: readonly Column<Row>[] = [
    {
      name: 'Time',
      key: 'time',
      resizable: true,
      width: '15%',
    },
    {
      name: 'Id',
      key: 'index',
      resizable: true,
      width: '15%',
    },
    {
      name: 'Answer',
      key: 'answer',
      resizable: true,
      width: '35%',
      renderCell: ({ row }) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
            userSelect: 'none',
          }}
        >
          {row.answer}
        </div>
      ),
    },
    {
      name: 'Prompt',
      key: 'prompt',
      resizable: true,
      width: '25%',
      renderCell: ({ row }) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
            userSelect: 'none',
            backgroundColor: '#f0f0f0',
          }}
        >
          {row.prompt}
        </div>
      ),
    },

    {
      name: 'Model name',
      key: 'model name',
      resizable: true,
      width: '10%',
    },
  ];

  const showAddForm = () => {
    setShowForm(true);
  };

  const hideForm = () => {
    setShowForm(false);
  };

  const addModel = async (model: Omit<GenModel & { api: SupportedAPI }, 'id'>) => {
    const id = await createGenModel(projectName, model);
    setConfigureModels([...configuredModels, { ...model, id }]);
    setShowForm(false);
  };

  const removeModel = async () => {
    setConfigureModels(configuredModels.filter((m) => m.id !== selectedModel));
    if (selectedModel !== undefined) await deleteGenModel(projectName, selectedModel);
  };

  const handleChange = async (e: ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(parseInt(e.target.value));
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="generate">
      <div className="container-fluid mt-3">
        <div className="row"></div>
        <div className="explanations">
          You can configure LLM-as-service to use prompt-engineering on your data
        </div>

        {showForm ? (
          <GenModelSetupForm add={addModel} cancel={hideForm} />
        ) : (
          <>
            {configuredModels.length === 0 ? (
              <>
                <p>No generative models assigned to this project</p>
                <AddButton showAddForm={showAddForm}></AddButton>
              </>
            ) : (
              <>
                <div className="row d-flex align-items-center">
                  <div className="col-6">
                    <div className="form-floating">
                      <select id="model" className="form-select" onChange={handleChange}>
                        {configuredModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name}
                          </option>
                        ))}
                      </select>
                      <label htmlFor="model">Select a model</label>
                    </div>
                  </div>
                  <div className="col-6">
                    <AddButton showAddForm={showAddForm}></AddButton>

                    <button onClick={removeModel} className="btn btn-primary mx-2">
                      <FaRegTrashAlt size={20} /> Delete
                    </button>
                  </div>
                </div>
                <hr />

                <div className="row mt-3">
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
                      <label htmlFor="mode">Select from </label>
                    </div>
                  </div>

                  <div className="explanations mt-3">
                    Select or craft your prompt with the element #INSERTTEXT to insert text
                  </div>

                  <div className="d-flex align-items-center ">
                    <Select
                      id="select-prompt"
                      className="form-select w-75"
                      options={(prompts || []).map((e) => ({
                        value: e.id as unknown as string,
                        label: e.text as unknown as string,
                      }))}
                      isClearable
                      placeholder="Look for a recorded prompt"
                      onChange={(e) => {
                        setAppContext((prev) => ({
                          ...prev,
                          generateConfig: {
                            ...generateConfig,
                            prompt: e?.label || '',
                            prompt_id: e?.value,
                          },
                        }));
                      }}
                    />
                    <button
                      className="btn btn-primary mx-2 savebutton"
                      onClick={() => {
                        savePrompts(generateConfig.prompt || null);
                        reFetchPrompts();
                      }}
                    >
                      <BsSave2 />
                    </button>
                    <Tooltip anchorSelect=".savebutton" place="top">
                      Save the prompt
                    </Tooltip>
                    <button
                      onClick={() => {
                        deletePrompts(generateConfig.prompt_id || null);
                        reFetchPrompts();
                      }}
                      className="btn btn-primary mx-2"
                    >
                      <FaRegTrashAlt size={20} />
                    </button>
                  </div>
                  <div className="form-floating mt-2">
                    <textarea
                      id="prompt"
                      rows={5}
                      placeholder="Enter your prompt"
                      className="form-control"
                      style={{ height: '200px', backgroundColor: '#fff0fe' }}
                      value={generateConfig.prompt || ''}
                      onChange={(e) => {
                        setAppContext((prev) => ({
                          ...prev,
                          generateConfig: { ...generateConfig, prompt: e.target.value },
                        }));
                      }}
                    />
                    <span style={{ color: 'gray' }}>
                      The request will send the data to the external API. Be sure you can trust the
                      API provider with the level of privacy of your data.
                    </span>
                    <label htmlFor="prompt">Prompt </label>
                  </div>
                  <div className="col-12 text-center">
                    {isGenerating ? (
                      <div>
                        <PulseLoader />
                        <button className="btn btn-secondary mt-3" onClick={stopGenerate}>
                          Stop
                        </button>
                      </div>
                    ) : (
                      <button
                        className="btn btn-secondary mt-3 generatebutton"
                        onClick={() => {
                          console.log('generate');
                          generate();
                        }}
                      >
                        <HiOutlineSparkles size={30} /> Generate
                      </button>
                    )}
                    <Tooltip anchorSelect=".generatebutton" place="top">
                      It can take some time if you have a large batch
                    </Tooltip>
                  </div>
                </div>
                <hr />
                <div className="col-12 d-flex align-items-center justify-content-center">
                  <span>Number elements to download</span>
                  <input
                    type="number"
                    placeholder="Number of last generated elements to download"
                    className="form-control m-4"
                    style={{ width: '100px' }}
                    value={numberElements || 10}
                    onChange={(e) => setNumberElements(Number(e.target.value))}
                  />
                  <button
                    className="btn btn-primary"
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
                  rowHeight={80}
                />
              </>
            )}
          </>
        )}
      </div>
    </ProjectPageLayout>
  );
};
