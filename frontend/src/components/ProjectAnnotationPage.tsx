import { FC, useCallback, useEffect, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Highlighter from 'react-highlight-words';
import { FaPencilAlt } from 'react-icons/fa';
import { IoMdSkipBackward } from 'react-icons/io';
import { LuRefreshCw } from 'react-icons/lu';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useStatistics,
  useUpdateSimpleModel,
} from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';
import { MulticlassInput } from './MulticlassInput';
import { MultilabelInput } from './MultilabelInput';
import { ProjectionManagement } from './ProjectionManagement';
import { SelectionManagement } from './SelectionManagement';
import { SimpleModelManagement } from './SimpleModelManagement';

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
      phase,
    },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element
  const [nSample, setNSample] = useState<number | null>(null); // specific info
  const [displayComment, setDisplayComment] = useState(false);
  const [comment, setComment] = useState('');

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
    if (elementId === 'noelement') {
      return;
    }
    if (elementId === undefined) {
      getNextElementId().then((res) => {
        if (res && res.n_sample) setNSample(res.n_sample);
        if (res && res.element_id) navigate(`/projects/${projectName}/annotate/${res.element_id}`);
        else {
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
    (label: string, elementId?: string) => {
      if (elementId) {
        setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
        addAnnotation(elementId, label, comment).then(() =>
          // redirect to next element by redirecting wihout any id
          // thus the getNextElementId query will be dont after the appcontext is reloaded
          {
            setComment(''); // reset comment
            navigate(`/projects/${projectName}/annotate/`); // got to next element
          },
        );
        // does not do nothing as we remount through navigate reFetchStatistics();
      }
    },
    [setAppContext, addAnnotation, navigate, projectName, comment],
  );

  const textInFrame = element?.text.slice(0, element?.limit as number) || '';
  const textOutFrame = element?.text.slice(element?.limit as number) || '';

  const lastTag =
    element?.history.length && element?.history.length > 0
      ? (element?.history[0] as string[])[0]
      : null;

  const refetchElement = () => {
    getNextElementId().then((res) => {
      if (res && res.n_sample) setNSample(res.n_sample);
      if (res && res.element_id) navigate(`/projects/${projectName}/annotate/${res.element_id}`);
      else {
        navigate(`/projects/${projectName}/annotate/noelement`);
      }
    });
  };

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="annotate">
      <div className="container-fluid">
        <div className="row mb-3 mt-3">
          {
            // test mode
            phase == 'test' && (
              <div className="alert alert-warning">
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
                    <span
                      className="badge text-bg-light"
                      title="tagged / selection untagged / total"
                    >
                      Annotated :{' '}
                      {`${statistics[phase == 'test' ? 'test_annotated_n' : 'train_annotated_n']} / ${nSample ? nSample : ''} / ${statistics[phase == 'test' ? 'test_set_n' : 'train_set_n']}`}
                    </span>
                  ) : (
                    ''
                  )}

                  <div>
                    <button className="btn" onClick={refetchElement}>
                      <LuRefreshCw size={20} /> Refetch element
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
            Refetch with current selection mode
          </button>
        </div>
      )}

      {
        // display content
      }
      <div className="row">
        {element?.text && (
          <div
            className="col-11 annotation-frame"
            style={{ height: `${displayConfig.frameSize}vh` }}
          >
            {lastTag && (
              <div>
                <span className="badge bg-info  ">Last tag: {lastTag}</span>
              </div>
            )}
            <Highlighter
              highlightClassName="Search"
              searchWords={
                selectionConfig.filter && selectionConfig.filter.slice(-1) != '\\'
                  ? [selectionConfig.filter]
                  : []
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
                  selectionConfig.filter && selectionConfig.filter.slice(-1) != '\\'
                    ? [selectionConfig.filter]
                    : []
                }
                autoEscape={false}
                textToHighlight={textOutFrame}
                caseSensitive={true}
              />
            </span>
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

      <div className="row">
        <div className="d-flex flex-wrap gap-2 justify-content-center">
          <Link
            to={`/projects/${projectName}/annotate/${history[history.length - 1]}`}
            className="btn btn-outline-secondary"
            onClick={() => {
              setAppContext((prev) => ({ ...prev, history: prev.history.slice(0, -1) }));
            }}
          >
            <IoMdSkipBackward />
          </Link>
          <button className="btn" onClick={() => setDisplayComment(!displayComment)}>
            <FaPencilAlt />
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

      <div className="mt-5">
        {phase != 'test' && (
          <Tabs id="panel2" className="mb-3">
            <Tab eventKey="prediction" title="Prediction">
              <SimpleModelManagement
                projectName={projectName || null}
                currentScheme={currentScheme || null}
                availableSimpleModels={availableSimpleModels}
                availableFeatures={availableFeatures}
              />
            </Tab>
            <Tab eventKey="visualization" title="Visualization">
              <ProjectionManagement currentElementId={elementId || null} />
            </Tab>

            <Tab eventKey="parameters" title="Display parameters">
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
                Display prediction
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
                Display informations
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
                Display annotation history
              </label>
              <label style={{ display: 'block', marginBottom: '10px' }}>
                Text frame size
                <span>Min: 25%</span>
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
            </Tab>
          </Tabs>
        )}
      </div>
    </ProjectPageLayout>
  );
};
