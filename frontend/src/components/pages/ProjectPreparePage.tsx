import { FC } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';

import { useAddFeature, useDeleteFeature, useGetFeatureInfo } from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { FeatureModelExtended } from '../../types';
import { ImportAnnotations } from '../ImportAnnotations';
import { LabelsManagement } from '../LabelsManagement';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

/**
 * Component to display the features page
 */
interface FastTextOptions {
  models?: string[];
}

interface FeaturesOptions {
  fasttext?: FastTextOptions;
}

interface Features {
  options?: FeaturesOptions;
}

interface FeatureComputingElement {
  name: string;
  progress: string | null;
}

export const ProjectPreparePage: FC = () => {
  const { projectName } = useParams();

  // get element from the state
  const {
    appContext: { currentProject: project, currentScheme, reFetchCurrentProject },
  } = useAppContext();

  // API calls
  const { featuresInfo } = useGetFeatureInfo(projectName || null, project);
  const addFeature = useAddFeature(projectName || null);
  const deleteFeature = useDeleteFeature(projectName || null);

  // hooks to use the objets
  const { register, handleSubmit, watch, reset } = useForm<FeatureModelExtended>({
    defaultValues: { parameters: { value: '' }, type: '' },
  });
  const { notify } = useNotifications();

  // state for the type of feature to create
  const selectedFeatureToCreate = watch('type');

  // action to create the new scheme
  const createNewFeature: SubmitHandler<FeatureModelExtended> = async (formData) => {
    try {
      addFeature(
        formData.type,
        formData.name,
        formData.parameters as unknown as Record<string, string | number | undefined>,
      );
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    reset();
  };

  const deleteSelectedFeature = async (element: string) => {
    await deleteFeature(element);
  };

  const availableLabels =
    currentScheme && project ? project.schemes.available[currentScheme]['labels'] || [] : [];
  const kindScheme =
    currentScheme && project ? project.schemes.available[currentScheme]['kind'] : '';

  console.log(project?.features.training);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="prepare">
      {project && projectName && (
        <div className="container-fluid">
          <div className="row">
            <div className="col-12">
              <Tabs id="panel" className="mt-3" defaultActiveKey="labels">
                <Tab eventKey="labels" title="Labels">
                  <LabelsManagement
                    projectName={projectName || null}
                    currentScheme={currentScheme || null}
                    availableLabels={availableLabels}
                    kindScheme={kindScheme}
                    reFetchCurrentProject={reFetchCurrentProject || (() => null)}
                  />
                </Tab>
                <Tab eventKey="features" title="Features">
                  <span className="explanations">Create and delete features.</span>
                  <h4 className="mt-3 subsection">Existing features</h4>
                  {/* Display existing features */}
                  {Object.entries(featuresInfo || {}).map(([key, value]) => (
                    <div className="card text-bg-light mt-3" key={key}>
                      <div className="d-flex m-2 align-items-center">
                        <button
                          className="btn btn p-0 mx-4"
                          onClick={() => {
                            deleteSelectedFeature(key);
                          }}
                        >
                          <MdOutlineDeleteOutline size={20} />
                        </button>
                        <span className="w-25">{key}</span>
                        <span className="mx-2">{value?.time}</span>
                        <span className="mx-2">by {value?.user}</span>
                        {value?.kind === 'regex' && <span>N={value.parameters['count']}</span>}
                      </div>
                    </div>
                  ))}{' '}
                  {/* Display computing features */}
                  {Object.entries(project?.features.training).map(([key, element]) => (
                    <div className="card text-bg-light mt-3 bg-warning" key={key}>
                      <div className="d-flex m-2 align-items-center">
                        <span className="w-25">
                          Currently computing {(element as FeatureComputingElement).name as string}
                          {(element as FeatureComputingElement).progress
                            ? ` (${(element as FeatureComputingElement).progress}%)`
                            : ''}
                        </span>
                      </div>
                    </div>
                  ))}
                  {/* // create new feature */}
                  <h4 className="mt-3 subsection">Create a new feature</h4>
                  <form onSubmit={handleSubmit(createNewFeature)}>
                    <span className="explanations">
                      Depending on the size of the corpus, computation can take some time (up to
                      dozens of minutes)
                    </span>
                    <select className="form-control" id="newFeature" {...register('type')}>
                      <option key="empty"></option>
                      {Object.keys(project.features.options).map((element) => (
                        <option key={element} value={element}>
                          {element}
                        </option>
                      ))}{' '}
                    </select>

                    {/* {selectedFeatureToCreate === 'sbert' && (
                      <div>
                        <label htmlFor="dfm_norm">Model to use</label>
                        <select id="dfm_norm" {...register('parameters.dfm_norm')}>
                          {(project?.features.options.sbert.models || []).map((element) => (
                            <option key={element as string} value={element as string}>
                              {element as string}
                            </option>
                          ))}
                        </select>
                      </div>
                    )} */}

                    {selectedFeatureToCreate === 'fasttext' && (
                      <div>
                        <label htmlFor="dataset_col">Optional, model to use</label>
                        <select id="dataset_col" {...register('parameters.model')}>
                          <option key={null} value={''}>
                            Generic model
                          </option>

                          {((project?.features as Features).options?.fasttext?.models || []).map(
                            (element) => (
                              <option key={element as string} value={element as string}>
                                {element as string}
                              </option>
                            ),
                          )}
                        </select>
                      </div>
                    )}

                    {selectedFeatureToCreate === 'regex' && (
                      <input
                        type="text"
                        className="form-control mt-3"
                        placeholder="Enter the regex"
                        {...register('parameters.value')}
                      />
                    )}

                    {selectedFeatureToCreate === 'dfm' && (
                      <div>
                        <div>
                          <label htmlFor="dfm_tfidf">TF-IDF</label>
                          <select id="dfm_tfidf" {...register('parameters.dfm_tfidf')}>
                            <option key="true">True</option>
                            <option key="false">False</option>
                          </select>
                        </div>
                        <div>
                          <label htmlFor="dfm_ngrams">Ngrams</label>
                          <input
                            type="number"
                            id="dfm_ngrams"
                            value={1}
                            {...register('parameters.dfm_ngrams')}
                          />
                        </div>
                        <div>
                          <label htmlFor="dfm_min_term_freq">Min term freq</label>
                          <input
                            type="number"
                            id="dfm_min_term_freq"
                            value={5}
                            {...register('parameters.dfm_min_term_freq')}
                          />
                        </div>
                        <div>
                          <label htmlFor="dfm_max_term_freq">Max term freq</label>
                          <input
                            type="number"
                            id="dfm_max_term_freq"
                            value={100}
                            {...register('parameters.dfm_max_term_freq')}
                          />
                        </div>
                        <div>
                          <label htmlFor="dfm_norm">Norm</label>
                          <select id="dfm_norm" {...register('parameters.dfm_norm')}>
                            <option key="true">True</option>
                            <option key="false">False</option>
                          </select>
                        </div>
                        <div>
                          <label htmlFor="dfm_log">Log</label>
                          <select id="dfm_log" {...register('parameters.dfm_log')}>
                            <option key="true">True</option>
                            <option key="false">False</option>
                          </select>
                        </div>
                      </div>
                    )}

                    {selectedFeatureToCreate === 'dataset' && (
                      <div>
                        <label htmlFor="dataset_col">Column to use</label>
                        <select id="dataset_col" {...register('parameters.dataset_col')}>
                          {(project?.params.all_columns || []).map((element) => (
                            <option key={element as string} value={element as string}>
                              {element as string}
                            </option>
                          ))}
                        </select>
                        <label htmlFor="dataset_type">Type of the feature</label>
                        <select id="dataset_type" {...register('parameters.dataset_type')}>
                          <option key="numeric">Numeric</option>
                          <option key="categorical">Categorical</option>
                        </select>
                      </div>
                    )}

                    <button className="btn btn-primary btn-validation">Create</button>
                  </form>
                </Tab>
                <Tab eventKey="imports" title="Import annotations">
                  <ImportAnnotations
                    projectName={projectName}
                    currentScheme={currentScheme || null}
                  />
                </Tab>
              </Tabs>
            </div>
          </div>
          <div></div>
        </div>
      )}
    </ProjectPageLayout>
  );
};
