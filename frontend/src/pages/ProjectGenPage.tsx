import { FC, useEffect, useState } from 'react';
import { Modal } from 'react-bootstrap';
import DataGrid, { Column } from 'react-data-grid';
import { BsSave2 } from 'react-icons/bs';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle, HiOutlineSparkles } from 'react-icons/hi';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';
import { GenModelSetupForm } from '../components/forms/GenModelSetupForm';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
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

interface Row {
  time: string;
  index: string;
  prompt: string;
  answer: string;
  endpoint: string;
}

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
  const [currentModel, setCurrentModel] = useState<string | null>(null);
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

  // get existing models and select the first
  useEffect(() => {
    const fetchModels = async () => {
      const models = await getProjectGenModels(projectName);
      setConfiguredModels(models);
    };
    fetchModels();
  }, [projectName, currentModel]);

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

  const deleteModel = async (name: string) => {
    const id = configuredModels.filter((m) => m.name === name)[0].id;
    await deleteGenModel(projectName, id).then(() => {
      setCurrentModel(null);
      notify({ type: 'success', message: 'Model removed' });
      return true;
    });
  };

  // function to handle the change of the model
  // const handleChange = async (e: ChangeEvent<HTMLSelectElement>) => {
  //   const model = configuredModels.filter((m) => m.id === parseInt(e.target.value))[0];
  //   setAppContext((prev) => ({
  //     ...prev,
  //     generateConfig: { ...prev.generateConfig, selectedModel: model },
  //   }));
  // };

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

  const addContextTagToPrompt = (context: string) => {
    setAppContext((prev) => ({
      ...prev,
      generateConfig: {
        ...generateConfig,
        prompt: (generateConfig.prompt =
          (generateConfig.prompt ? generateConfig.prompt : '') + '[[' + context + ']]'),
      },
    }));
  };
  const addContextButtons = (contextColumns: string[] | undefined) => {
    let listOfTags: string[] = ['TEXT'];
    if (contextColumns) listOfTags = listOfTags.concat(contextColumns);

    return (
      <>
        <div className="col-12" id="context-container">
          <div id="button-context-wrapper">
            <a className="context-tag-help">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".context-tag-help" place="top" style={{ zIndex: 99 }}>
              Add contextual information to your prompt by clicking on the tags, or by typing
              [[column name]]
            </Tooltip>
            {listOfTags.map((context) => (
              <button
                className="context-link"
                id="context-button"
                key={'add-context-button-' + context}
                onClick={() => addContextTagToPrompt(context)}
              >
                {context}
              </button>
            ))}
          </div>
        </div>
      </>
    );
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="generate">
      <div className="container-fluid mt-3">
        <div className="row"></div>
        <div className="explanations">Use external LLM models for generation</div>
        <Modal show={showForm} id="createmodel-modal" size="xl" onHide={() => setShowForm(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Add a new generative model</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <GenModelSetupForm add={addModel} cancel={hideForm} />
          </Modal.Body>
        </Modal>
        <ModelsPillDisplay
          modelNames={configuredModels.map((m) => m.name)}
          currentModelName={currentModel}
          setCurrentModelName={setCurrentModel}
          deleteModelFunction={deleteModel}
        >
          <button onClick={showAddForm} className="model-pill" id="create-new">
            <FaPlusCircle size={20} /> Add new model
          </button>
        </ModelsPillDisplay>
        {currentModel && (
          <>
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
                Select or craft your prompt with the element [[TEXT]] to insert text
              </div>

              <div className="d-flex align-items-center " style={{ zIndex: 1 }}>
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
                    <Tooltip anchorSelect=".savebutton" place="top" style={{ zIndex: 99 }}>
                      Save the prompt
                    </Tooltip>
                  </div>
                </details>
              </div>
              {addContextButtons(currentProject?.params.cols_context)}
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
                  The request will send the data to an external API. Be sure you can trust the API
                  provider with respect to the level of privacy you need for you data
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
                        currentProject?.generations?.training?.[authenticatedUser?.username || '']
                          ?.progress ?? 0,
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
            <div className="col-12 d-flex align-items-center justify-content-between">
              <h4 className="subsection">Results</h4>
              <div>
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
      </div>
    </ProjectPageLayout>
  );
};
