import { FC, useCallback, useEffect, useRef, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { FaPencilAlt } from 'react-icons/fa';
import { LuRefreshCw } from 'react-icons/lu';
import { PiEraser } from 'react-icons/pi';
import { useNavigate, useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { BackButton } from '../components/BackButton';
import { ForwardButton } from '../components/ForwardButton';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useStatistics,
} from '../core/api';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';

import { MdDisplaySettings } from 'react-icons/md';
import { useLocation } from 'react-router-dom';
import { ActiveLearningManagement } from '../components/ActiveLearningManagement';
import { AnnotationDisagreementManagement } from '../components/AnnotationDisagreementManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { MulticlassInput } from '../components/MulticlassInput';
import { MultilabelInput } from '../components/MultilabelInput';
import { SchemesComparisonManagement } from '../components/SchemesComparisonManagement';
import { SelectionManagement } from '../components/SelectionManagement';
import { TagDisplayParameters } from '../components/TagDisplayParameters';
import { TextClassificationPanel } from '../components/TextClassificationPanel';
import { TextSpanPanel } from '../components/TextSpanPanel';

/**
 * Annotation page
 */
export const ProjectTagPage: FC = () => {
  // parameters
  const { projectName, elementId } = useParams();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const tab = queryParams.get('tab');
  const {
    appContext: {
      currentScheme,
      currentProject: project,
      selectionConfig,
      displayConfig,
      freqRefreshSimpleModel,
      activeSimpleModel,
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
  const [activeTab, setActiveTab] = useState<string>('tag');
  const [showDisplayConfig, setShowDisplayConfig] = useState<boolean>(false);
  useEffect(() => {
    setActiveTab(tab || 'tag');
  }, [tab]);

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
    activeSimpleModel || null,
  );
  const { getElementById } = useGetElementById(
    projectName || null,
    currentScheme || null,
    activeSimpleModel || null,
  );

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, phase);

  // define parameters for configuration panels
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
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
          navigate(`/projects/${projectName}/tag/${res.element_id}`);
        } else {
          navigate(`/projects/${projectName}/tag/noelement`);
          setElement(null);
        }
      });
    } else {
      getElementById(elementId, phase).then((element) => {
        if (element) setElement(element);
        else {
          navigate(`/projects/${projectName}/tag/noelement`);
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
            navigate(`/projects/${projectName}/tag/`); // got to next element
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
      if (res && res.n_sample) setNSample(res.n_sample);
      if (res && res.element_id) navigate(`/projects/${projectName}/tag/${res.element_id}`);
      else {
        navigate(`/projects/${projectName}/tag/noelement`);
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
  const highlightTextRaw = [selectionConfig.filter, ...displayConfig.highlightText.split('\n')];
  const highlightText = highlightTextRaw.filter(
    (text): text is string => typeof text === 'string' && text.trim() !== '',
  );

  // Now filter by valid regex
  const validHighlightText = highlightText.filter(isValidRegex);

  //display switch to test mode
  const isTest = statistics?.test_set_n ? statistics?.test_set_n > 0 : false;
  const isValid = statistics?.valid_set_n ? statistics?.valid_set_n > 0 : false;

  // existing simplemodels
  const availableSimpleModels = project?.simplemodel.available[currentScheme || ''] || [];

  if (!projectName || !currentScheme) return;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="tag">
      <Tabs className="mt-3" activeKey={activeTab} onSelect={(k) => setActiveTab(k || 'tag')}>
        <Tab eventKey="tag" title="Tag">
          <div className="container-fluid">
            <div className="row mb-3 mt-3">
              {
                // annotation mode
                <div>
                  <div
                    className={`d-flex align-items-center mb-3 ${phase !== 'train' ? 'alert alert-warning' : ''}`}
                  >
                    <div className="m-2">
                      <div className="col-12 d-flex justify-content-center align-items-center">
                        Dataset{' '}
                        <select
                          className="form-select mx-3"
                          value={phase}
                          onChange={(e) => {
                            setAppContext((prev) => ({
                              ...prev,
                              phase: e.target.value,
                            }));
                            navigate(`/projects/${projectName}/tag/`);
                          }}
                        >
                          <option value="train">Train</option>
                          {isValid && <option value="valid">Validation</option>}
                          {isTest && <option value="test">Test</option>}
                        </select>
                      </div>
                    </div>
                    {statistics ? (
                      <span className="badge text-bg-light currentstatistics">
                        Annotated :{' '}
                        {`${statistics[phase == 'test' ? 'test_annotated_n' : 'train_annotated_n']} / ${statistics[phase == 'test' ? 'test_set_n' : 'train_set_n']} ; Selected : ${nSample ? nSample : ''} `}
                      </span>
                    ) : (
                      ''
                    )}
                    <Tooltip anchorSelect=".currentstatistics" place="top">
                      statistics for the current scheme
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
              }
            </div>
          </div>
          {kindScheme !== 'span' ? (
            <>
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

              <TextClassificationPanel
                element={element as ElementOutModel}
                displayConfig={displayConfig}
                textInFrame={textInFrame}
                textOutFrame={textOutFrame}
                validHighlightText={validHighlightText}
                elementId={elementId as string}
                lastTag={lastTag as string}
                phase={phase}
                frameRef={frameRef as unknown as HTMLDivElement}
                postAnnotation={postAnnotation}
              />

              {showDisplayConfig && (
                <TagDisplayParameters displayConfig={displayConfig} setAppContext={setAppContext} />
              )}
            </>
          ) : (
            <>
              <TextSpanPanel
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
                text={element?.text as string}
                lastTag={lastTag as string}
              />
            </>
          )}
          {elementId !== 'noelement' && (
            <div className="row">
              <div className="d-flex flex-wrap gap-2 justify-content-center">
                <BackButton
                  projectName={projectName || ''}
                  history={history}
                  setAppContext={setAppContext}
                />

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
            </div>
          )}
          <div className="d-flex flex-wrap gap-2 justify-content-center">
            <button className="btn addcomment" onClick={() => setDisplayComment(!displayComment)}>
              <FaPencilAlt />
              <Tooltip anchorSelect=".addcomment" place="top">
                Add a commentary
              </Tooltip>
            </button>

            <button
              className="btn displayconfig"
              onClick={() => {
                setShowDisplayConfig(!showDisplayConfig);
              }}
            >
              <MdDisplaySettings />
              <Tooltip anchorSelect=".displayconfig" place="top">
                Display config menu
              </Tooltip>
            </button>
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
        </Tab>
        <Tab eventKey="active" title="Active">
          <ActiveLearningManagement
            projectSlug={projectName}
            history={history}
            currentScheme={currentScheme}
            availableSimpleModels={availableSimpleModels}
            setAppContext={setAppContext}
            freqRefreshSimpleModel={freqRefreshSimpleModel}
            activeSimepleModel={activeSimpleModel}
          />
        </Tab>
        <Tab eventKey="curate" title="Curate">
          <Tabs id="panel" className="mt-3" defaultActiveKey="scheme">
            <Tab eventKey="scheme" title="Current scheme">
              <AnnotationDisagreementManagement projectSlug={projectName} />
            </Tab>
            <Tab eventKey="between" title="Between schemes">
              <SchemesComparisonManagement projectSlug={projectName} />
            </Tab>
          </Tabs>
        </Tab>
      </Tabs>
    </ProjectPageLayout>
  );
};
