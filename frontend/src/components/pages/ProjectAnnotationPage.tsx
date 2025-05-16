import { motion } from 'framer-motion';
import { FC, useCallback, useEffect, useRef, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Highlighter from 'react-highlight-words';
import { FaPencilAlt } from 'react-icons/fa';
import { LuRefreshCw } from 'react-icons/lu';
import { PiEraser } from 'react-icons/pi';
import { useNavigate, useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useStatistics,
  useUpdateSimpleModel,
} from '../../core/api';
import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';
import { ElementOutModel } from '../../types';
import { BackButton } from '../BackButton';
import { ForwardButton } from '../ForwardButton';

import { ProjectPageLayout } from '../layout/ProjectPageLayout';
import { MulticlassInput } from '../MulticlassInput';
import { MultilabelInput } from '../MultilabelInput';
import { ProjectionManagement } from '../ProjectionManagement';
import { SelectionManagement } from '../SelectionManagement';
import { SimpleModelManagement } from '../SimpleModelManagement';

/**
 * Annotation page
 */
export const ProjectAnnotationPage: FC = () => {
  // parameters
  const { projectName, elementId } = useParams();
  const { authenticatedUser } = useAuth();
  const {
    appContext: {
      currentScheme,
      currentProject: project,
      selectionConfig,
      displayConfig,
      freqRefreshSimpleModel,
      history,
      selectionHistory,
      phase,
    },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element
  const [nSample, setNSample] = useState<number | null>(null); // specific info
  const [displayComment, setDisplayComment] = useState(false);
  const [comment, setComment] = useState('');

  // Reinitialize scroll in frame
  const frameRef = useRef<HTMLDivElement>(null);
  const resetScroll = () => {
    if (frameRef.current) {
      frameRef.current.scrollTop = 0;
    }
  };

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(
    projectName || null,
    currentScheme || null,
    selectionConfig,
    history,
    phase,
  );
  const { getElementById } = useGetElementById(projectName || null, currentScheme || null);

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, phase);

  // define parameters for configuration panels
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableSimpleModels = project?.simplemodel.options ? project?.simplemodel.options : {};
  const currentModel =
    authenticatedUser &&
    currentScheme &&
    project?.simplemodel.available[authenticatedUser?.username]?.[currentScheme]
      ? project?.simplemodel.available[authenticatedUser?.username][currentScheme]
      : null;

  const availableLabels =
    currentScheme && project
      ? (project.schemes.available[currentScheme]['labels'] as string[])
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project
      ? (project.schemes.available[currentScheme]['kind'] as string) || 'multiclass'
      : 'multiclass',
  );

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics, reFetchStatistics } = useStatistics(
    projectName || null,
    currentScheme || null,
  );

  // react to URL param change
  useEffect(() => {
    resetScroll();
    if (elementId === 'noelement') {
      return;
    }
    if (elementId === undefined) {
      getNextElementId().then((res) => {
        if (res && res.n_sample) setNSample(res.n_sample);
        if (res && res.element_id) {
          setAppContext((prev) => ({
            ...prev,
            selectionHistory: {
              ...prev.selectionHistory,
              [res.element_id]: JSON.stringify(selectionConfig),
            },
          }));
          navigate(`/projects/${projectName}/annotate/${res.element_id}`);
        } else {
          navigate(`/projects/${projectName}/annotate/noelement`);
          setElement(null);
        }
      });
    } else {
      getElementById(elementId, phase).then((element) => {
        if (element) setElement(element);
        else {
          navigate(`/projects/${projectName}/annotate/noelement`);
          setElement(null);
        }
      });
      reFetchStatistics();
    }
  }, [
    elementId,
    getNextElementId,
    getElementById,
    navigate,
    phase,
    projectName,
    reFetchStatistics,
    selectionConfig,
    setAppContext,
  ]);

  // hooks to update simplemodel
  const [updatedSimpleModel, setUpdatedSimpleModel] = useState(false); // use a memory to only update once
  const { updateSimpleModel } = useUpdateSimpleModel(projectName || null, currentScheme || null);

  useEffect(() => {
    // conditions to update the model
    if (
      !updatedSimpleModel &&
      currentModel &&
      history.length > 0 &&
      history.length % freqRefreshSimpleModel == 0
    ) {
      setUpdatedSimpleModel(true);
      updateSimpleModel(currentModel);
    }
    if (updatedSimpleModel && history.length % freqRefreshSimpleModel != 0)
      setUpdatedSimpleModel(false);
  }, [
    history,
    updateSimpleModel,
    setUpdatedSimpleModel,
    currentModel,
    freqRefreshSimpleModel,
    updatedSimpleModel,
  ]);

  // post an annotation
  const postAnnotation = useCallback(
    (label: string | null, elementId?: string) => {
      if (elementId === 'noelement') return; // forbid annotation on noelement
      if (elementId) {
        addAnnotation(elementId, label, comment, selectionHistory[elementId]).then(() =>
          // redirect to next element by redirecting wihout any id
          // thus the getNextElementId query will be dont after the appcontext is reloaded
          {
            setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
            setComment('');
            navigate(`/projects/${projectName}/annotate/`); // got to next element
          },
        );
        // does not do nothing as we remount through navigate reFetchStatistics();
      }
    },
    [setAppContext, addAnnotation, navigate, projectName, comment, selectionHistory],
  );

  const textInFrame = element?.text.slice(0, displayConfig.numberOfTokens * 4) || '';
  const textOutFrame = element?.text.slice(displayConfig.numberOfTokens * 4) || '';

  const lastTag =
    element?.history?.length && element?.history.length > 0
      ? (element?.history[0] as string[])[0]
      : null;

  const refetchElement = () => {
    getNextElementId().then((res) => {
      console.log('res', res);
      if (res && res.n_sample) setNSample(res.n_sample);
      if (res && res.element_id) navigate(`/projects/${projectName}/annotate/${res.element_id}`);
      else {
        navigate(`/projects/${projectName}/annotate/noelement`);
      }
    });
  };

  const isValidRegex = (pattern: string) => {
    try {
      new RegExp(pattern);
      return true;
    } catch (e) {
      return false;
    }
  };

  console.log('history', history);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="annotate">
      <div className="container-fluid">
        <div className="row mb-3 mt-3">
          {
            // test mode
            phase == 'test' && (
              <div className="alert alert-info">
                Test mode activated - you are annotating the test set
                <div className="col-6">
                  {statistics && (
                    <span className="badge text-bg-light  m-3">
                      Number of annotations :{' '}
                      {`${statistics['test_annotated_n']} / ${statistics['test_set_n']}`}
                    </span>
                  )}
                </div>
              </div>
            )
          }
          {
            // annotation mode
            phase != 'test' && (
              <div>
                <div className="d-flex align-items-center mb-3">
                  {statistics ? (
                    <span className="badge text-bg-light currentstatistics">
                      Annotated :{' '}
                      {`${statistics[phase == 'test' ? 'test_annotated_n' : 'train_annotated_n']} / ${nSample ? nSample : ''} / ${statistics[phase == 'test' ? 'test_set_n' : 'train_set_n']}`}
                    </span>
                  ) : (
                    ''
                  )}
                  <Tooltip anchorSelect=".currentstatistics" place="top">
                    tagged / sample selected / total
                  </Tooltip>
                  <div>
                    <button className="btn getelement" onClick={refetchElement}>
                      <LuRefreshCw size={20} /> Get element
                      <Tooltip anchorSelect=".getelement" place="top">
                        Get next element with the selection mode
                      </Tooltip>
                    </button>
                  </div>
                </div>
                <div>
                  <SelectionManagement />
                </div>
              </div>
            )
          }
        </div>
      </div>

      {elementId === 'noelement' && (
        <div className="alert alert-warning text-center">
          <div className="m-2">No element available</div>
          <button className="btn btn-primary" onClick={refetchElement}>
            Get element
          </button>
        </div>
      )}

      {
        // display content
      }

      {!isValidRegex(selectionConfig.filter || '') && (
        <div className="alert alert-danger">Regex not valid</div>
      )}

      <div className="row">
        {element?.text && (
          <div
            className="col-11 annotation-frame"
            style={{ height: `${displayConfig.frameSize}vh` }}
            ref={frameRef}
          >
            <motion.div
              animate={elementId ? { backgroundColor: ['#e8e9ff', '#f9f9f9'] } : {}}
              transition={{ duration: 1 }}
            >
              {lastTag && (
                <div>
                  <span className="badge bg-info  ">
                    {displayConfig.displayAnnotation ? `Last tag: ${lastTag}` : 'Already annotated'}
                  </span>
                </div>
              )}

              <Highlighter
                highlightClassName="Search"
                searchWords={
                  selectionConfig.filter && isValidRegex(selectionConfig.filter)
                    ? [selectionConfig.filter, ...displayConfig.highlightText.split('\n')]
                    : displayConfig.highlightText.split('\n')
                }
                autoEscape={false}
                textToHighlight={textInFrame}
                highlightStyle={{
                  backgroundColor: 'yellow',
                  margin: '0px',
                  padding: '0px',
                }}
                caseSensitive={true}
              />
              {/* text out of frame */}
              <span className="text-out-context" title="Outside 512 token window ">
                <Highlighter
                  highlightClassName="Search"
                  searchWords={
                    selectionConfig.filter && isValidRegex(selectionConfig.filter)
                      ? [selectionConfig.filter, ...displayConfig.highlightText.split('\n')]
                      : []
                  }
                  autoEscape={false}
                  textToHighlight={textOutFrame}
                  caseSensitive={true}
                />
              </span>
            </motion.div>
          </div>
        )}

        {
          //display proba

          phase != 'test' && displayConfig.displayPrediction && element?.predict.label && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              {/* Predicted label : {element?.predict.label} (proba: {element?.predict.proba}) */}

              <button
                type="button"
                key={element?.predict.label + '_predict'}
                value={element?.predict.label}
                className="btn btn-secondary"
                onClick={(e) => {
                  postAnnotation(e.currentTarget.value, elementId);
                }}
              >
                Predicted : {element?.predict.label} (proba: {element?.predict.proba})
              </button>
            </div>
          )
        }
        {
          //display context
          phase != 'test' && displayConfig.displayContext && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              Context{' '}
              {Object.entries(element?.context || { None: 'None' }).map(
                ([k, v]) => `[${k} - ${v}]`,
              )}
            </div>
          )
        }
        {
          //display history
          phase != 'test' && displayConfig.displayHistory && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              {/* History : {JSON.stringify(element?.history)} */}
              History : {((element?.history as string[]) || []).map((h) => `[${h[0]} - ${h[2]}]`)}
            </div>
          )
        }
      </div>

      {elementId !== 'noelement' && (
        <div className="row">
          <div className="d-flex flex-wrap gap-2 justify-content-center">
            <BackButton
              projectName={projectName || ''}
              history={history}
              setAppContext={setAppContext}
            />

            <button className="btn addcomment" onClick={() => setDisplayComment(!displayComment)}>
              <FaPencilAlt />
              <Tooltip anchorSelect=".addcomment" place="top">
                Add a commentary
              </Tooltip>
            </button>

            {kindScheme == 'multiclass' && (
              <MulticlassInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
              />
            )}
            {kindScheme == 'multilabel' && (
              <MultilabelInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
              />
            )}
            {
              // erase button to remove last annotation
              lastTag && (
                <button
                  className="btn clearannotation"
                  onClick={() => {
                    postAnnotation(null, elementId);
                  }}
                >
                  <PiEraser />
                  <Tooltip anchorSelect=".clearannotation" place="top">
                    Erase current tag
                  </Tooltip>
                </button>
              )
            }
            {elementId && (
              <ForwardButton
                setAppContext={setAppContext}
                elementId={elementId}
                refetchElement={refetchElement}
              />
            )}
          </div>

          {displayComment && (
            <div className="m-3">
              <input
                type="text"
                className="form-control"
                placeholder="Comment"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
            </div>
          )}
        </div>
      )}

      <div className="mt-5">
        {phase != 'test' && (
          <Tabs id="panel2" className="mb-3">
            <Tab eventKey="prediction" title="Prediction">
              <SimpleModelManagement
                projectName={projectName || null}
                currentScheme={currentScheme || null}
                availableSimpleModels={availableSimpleModels}
                availableFeatures={availableFeatures}
                availableLabels={availableLabels}
                kindScheme={kindScheme}
              />
            </Tab>
            <Tab eventKey="visualization" title="Visualization" unmountOnExit={true}>
              <ProjectionManagement
                projectName={projectName || null}
                currentScheme={currentScheme || null}
                availableFeatures={availableFeatures}
              />
            </Tab>

            <Tab eventKey="parameters" title="Display parameters">
              <label style={{ display: 'block', marginBottom: '10px' }}>
                <input
                  type="checkbox"
                  checked={displayConfig.displayAnnotation}
                  onChange={(_) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        displayAnnotation: !displayConfig.displayAnnotation,
                      },
                    }));
                  }}
                  style={{ marginRight: '10px' }}
                />
                Existing annotation
              </label>
              <label style={{ display: 'block', marginBottom: '10px' }}>
                <input
                  type="checkbox"
                  checked={displayConfig.displayPrediction}
                  onChange={(_) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        displayPrediction: !displayConfig.displayPrediction,
                      },
                    }));
                  }}
                  style={{ marginRight: '10px' }}
                />
                Prediction
              </label>
              <label style={{ display: 'block', marginBottom: '10px' }}>
                <input
                  type="checkbox"
                  checked={displayConfig.displayContext}
                  onChange={(_) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        displayContext: !displayConfig.displayContext,
                      },
                    }));
                  }}
                  style={{ marginRight: '10px' }}
                />
                Contextual information
              </label>
              <label style={{ display: 'block', marginBottom: '10px' }}>
                <input
                  type="checkbox"
                  checked={displayConfig.displayHistory}
                  onChange={(_) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        displayHistory: !displayConfig.displayHistory,
                      },
                    }));
                  }}
                  style={{ marginRight: '10px' }}
                />
                Element history
              </label>
              <label style={{ display: 'block', marginBottom: '10px' }}>
                Tokens approximation {displayConfig.numberOfTokens} (4 c / token)
                <span className="m-2">Min: 100</span>
                <input
                  type="range"
                  min="100"
                  max="10000"
                  className="form-input"
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        numberOfTokens: Number(e.target.value),
                      },
                    }));
                  }}
                  style={{ marginRight: '10px' }}
                />
                <span>Max: 10000</span>
              </label>
              <label style={{ display: 'block', marginBottom: '10px' }}>
                Text frame size
                <span className="m-2">Min: 25%</span>
                <input
                  type="range"
                  min="25"
                  max="100"
                  className="form-input"
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        frameSize: Number(e.target.value),
                      },
                    }));
                  }}
                  style={{ marginRight: '10px' }}
                />
                <span>Max: 100%</span>
              </label>
              <div className="flex flex-col gap-2">
                <label className="explanations">Highlight words in the text</label>
                <br></br>
                <textarea
                  className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  placeholder="Line break to separate"
                  // onChange={(e) => setWordsToHighlight(e.target.value)}
                  value={displayConfig.highlightText}
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      displayConfig: {
                        ...displayConfig,
                        highlightText: String(e.target.value),
                      },
                    }));
                  }}
                />
              </div>
            </Tab>
          </Tabs>
        )}
      </div>
    </ProjectPageLayout>
  );
};
