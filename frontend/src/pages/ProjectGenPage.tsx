import { ChangeEvent, FC, useEffect, useState } from 'react';
import DataGrid, { Column } from 'react-data-grid';
import { BsSave2 } from 'react-icons/bs';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineSparkles } from 'react-icons/hi';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';
import { GenModelSetupForm } from '../components/forms/GenModelSetupForm';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import {
  createGenModel,
  deleteGenModel,
  getProjectGenModels,
  useDeletePrompts,
  useDropGeneratedElements,
  useGenerate,
  useGeneratedElements,
  useGetGenerationsFile,
  useGetPrompts,
  useSavePrompts,
  useStopGenerate,
} from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { GenModel, SupportedAPI } from '../types';

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
  //------------------------------------
  // hooks for the app
  const { projectName } = useParams() as { projectName: string };
  const { authenticatedUser } = useAuth();
  const {
    appContext: { generateConfig, currentScheme, currentProject },
    setAppContext,
  } = useAppContext();
  const { notify } = useNotifications();

  //------------------------------------
  // state of the page
  // genModels
  const [configuredModels, setConfiguredModels] = useState<Array<GenModel & { api: string }>>([]);
  const [showForm, setShowForm] = useState<boolean>(false);
  // currently generating for the user
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  // add filters for text treatment
  const [filters, setFilters] = useState<string[]>([]);
  // to save the prompt name
  const [promptName, setPromptName] = useState<string>('');

  //------------------------------------
  // call api
  // to post generation
  const { generate } = useGenerate(
    projectName || null,
    currentScheme || null,
    generateConfig.selectedModel?.id || null,
    generateConfig.n_batch || null,
    generateConfig.prompt || null,
    generateConfig.selectionMode || null,
    generateConfig.token,
  );
  // to stop generation
  const { stopGenerate } = useStopGenerate(projectName || null);

  // to get a sample of elements
  const { generated, reFetchGenerated } = useGeneratedElements(
    projectName || null,
    100,
    filters,
    isGenerating,
  );

  // to download a batch of elements
  const { getGenerationsFile } = useGetGenerationsFile(projectName || null, filters);

  // to drop generated elements
  const dropGeneratedElements = useDropGeneratedElements(
    projectName || null,
    authenticatedUser?.username || null,
  );

  // to get/save/delete prompts
  const { prompts, reFetchPrompts } = useGetPrompts(projectName || null);
  const savePrompts = useSavePrompts(projectName || null);
  const deletePrompts = useDeletePrompts(projectName || null);

  //------------------------------------
  // reacting elements

  // check if the user is generating and change the state
  useEffect(() => {
    setIsGenerating(
      authenticatedUser?.username !== undefined &&
        currentProject?.generations.training[authenticatedUser?.username] != undefined,
    );
  }, [authenticatedUser, currentProject]);

  // useEffect(() => {
  //   if (!generateConfig.api)
  //     setAppContext((prev) => ({
  //       ...prev,
  //       generateConfig: { ...generateConfig, api: 'ollama' },
  //     }));
  // }, [generateConfig, setAppContext]);

  // console.log('gen config', generateConfig);

  // get existing models and select the first
  useEffect(() => {
    const fetchModels = async () => {
      const models = await getProjectGenModels(projectName);
      setConfiguredModels(models);
      if (models.length > 0)
        setAppContext((prev) => ({
          ...prev,
          generateConfig: { ...prev.generateConfig, selectedModel: models[0] },
        }));
    };
    fetchModels();
  }, [projectName, setAppContext]);

  // utility functions for the DOM
  const showAddForm = () => {
    setShowForm(true);
  };

  const hideForm = () => {
    setShowForm(false);
  };

  // function to add a model
  const addModel = async (model: Omit<GenModel & { api: SupportedAPI }, 'id'>) => {
    const id = await createGenModel(projectName, model);
    notify({ type: 'success', message: 'Model added' });
    setConfiguredModels([...configuredModels, { ...model, id }]);
    setShowForm(false);
  };

  // function to remove a model
  const removeModel = async () => {
    setConfiguredModels(configuredModels.filter((m) => m.id !== generateConfig.selectedModel?.id));
    if (generateConfig.selectedModel?.id !== undefined)
      await deleteGenModel(projectName, generateConfig.selectedModel?.id);
    notify({ type: 'success', message: 'Model removed' });
  };

  // function to handle the change of the model
  const handleChange = async (e: ChangeEvent<HTMLSelectElement>) => {
    const model = configuredModels.filter((m) => m.id === parseInt(e.target.value))[0];
    setAppContext((prev) => ({
      ...prev,
      generateConfig: { ...prev.generateConfig, selectedModel: model },
    }));
  };

  // Define the table
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
                {/* <div>{JSON.stringify(generateConfig.selectedModel, null, 2)}</div> */}
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
                            generateConfig: { ...generateConfig, selectionMode: e.target.value },
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

                  <div className="d-flex align-items-center " style={{ zIndex: 100 }}>
                    <Select
                      id="select-prompt"
                      className="w-75"
                      options={(prompts || []).map((e) => ({
                        value: e.id as unknown as string,
                        label: e.parameters.name as unknown as string,
                        text: e.text as unknown as string,
                      }))}
                      isClearable
                      placeholder="Look for a recorded prompt"
                      onChange={(e) => {
                        setAppContext((prev) => ({
                          ...prev,
                          generateConfig: {
                            ...generateConfig,
                            prompt: e?.text || '',
                            promptId: e?.value,
                          },
                        }));
                      }}
                    />

                    <button
                      onClick={() => {
                        deletePrompts(generateConfig.promptId || null);
                        reFetchPrompts();
                      }}
                      className="btn btn-primary mx-2"
                    >
                      <FaRegTrashAlt size={20} />
                    </button>
                  </div>
                  <div>
                    {' '}
                    <details className="p-1  col-6">
                      <summary>Save prompt</summary>
                      <div className="d-flex align-items-center">
                        <input
                          type="text"
                          id="promptname"
                          className="form-control"
                          value={promptName}
                          placeholder="Prompt name to save"
                          onChange={(e) => setPromptName(e.target.value)}
                        />

                        <button
                          className="btn btn-primary mx-2 savebutton"
                          onClick={() => {
                            savePrompts(generateConfig.prompt || null, promptName);
                            reFetchPrompts();
                          }}
                        >
                          <BsSave2 />
                        </button>
                        <Tooltip anchorSelect=".savebutton" place="top">
                          Save the prompt
                        </Tooltip>
                      </div>
                    </details>
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
                      The request will send the data to an external API. Be sure you can trust the
                      API provider with respect to the level of privacy you need for you data
                    </span>
                    <label htmlFor="prompt" style={{ zIndex: 1 }}>
                      Prompt
                    </label>
                  </div>
                  <div className="col-12 text-center">
                    {isGenerating ? (
                      <div>
                        <div>
                          <PulseLoader />
                        </div>
                        <button className="btn btn-secondary mt-3" onClick={stopGenerate}>
                          Stop (
                          {String(
                            currentProject?.generations?.training?.[
                              authenticatedUser?.username || ''
                            ]?.progress ?? 0,
                          )}
                          % )
                        </button>
                      </div>
                    ) : (
                      <button
                        className="btn btn-secondary mt-3 generatebutton"
                        onClick={() => {
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
                  <button className="btn btn-primary mx-2" onClick={() => getGenerationsFile()}>
                    Download all
                  </button>
                  <button
                    className="btn btn-primary mx-2"
                    onClick={() => {
                      dropGeneratedElements().then(() => reFetchGenerated());
                    }}
                  >
                    Clear all
                  </button>
                </div>
                <Select
                  placeholder="Add treatment for the generated columns"
                  className="m-3"
                  options={[
                    { value: 'remove_punct', label: 'Remove punctuation' },
                    { value: 'remove_spaces', label: 'Remove spaces' },
                    { value: 'lowercase', label: 'Lowercase' },
                    { value: 'strip', label: 'Strip' },
                    { value: 'replace_accents', label: 'Replace accents characters' },
                  ]}
                  isMulti
                  onChange={(e) => {
                    setFilters(e.map((f) => f.value));
                  }}
                />
                <div className="explanations">Last 100 generated content for the current user</div>
                <DataGrid
                  className="fill-grid"
                  style={{ backgroundColor: 'white' }}
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
