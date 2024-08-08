import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';

import { useAddFeature, useDeleteFeature } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { FeatureModelExtended } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the features page
 */

export const ProjectFeaturesPage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;

  // get element from the state
  const {
    appContext: { currentProject: project },
  } = useAppContext();

  const availableFeatures = project ? Object.values(project.features.available) : [];

  // hooks to use the objets
  const { register, handleSubmit } = useForm<FeatureModelExtended>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const addFeature = useAddFeature(projectName);
  const deleteFeature = useDeleteFeature(projectName);

  // state for displaying the new scheme menu
  const [showCreateNewFeature, setShowCreateNewFeature] = useState(false);

  // state for the type of feature to create
  const [selectedFeatureToCreate, setFeatureToCreate] = useState('');

  // action to create the new scheme
  const createNewFeature: SubmitHandler<FeatureModelExtended> = async (formData) => {
    try {
      addFeature(formData.type, formData.name, formData.parameters);
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    //reFetch();
    setShowCreateNewFeature(!showCreateNewFeature);
  };

  // action to delete feature
  const deleteSelectedFeature = async (element: string) => {
    //TODO: try catch and throw
    await deleteFeature(element);
    //reFetch();
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="features">
      {project && (
        <div className="container-fluid">
          <div className="row">
            <h2 className="subsection">Managing features</h2>
            <span className="explanations">Features are computed from the textual data</span>

            {/*Display button to add features*/}
            <div className="row">
              <div className="col-3">
                <button
                  className="add-feature"
                  onClick={() => {
                    setShowCreateNewFeature(!showCreateNewFeature);
                  }}
                >
                  <IoIosAddCircle size={20} /> Add feature{' '}
                </button>
              </div>
            </div>
            {
              /*Display the menu to add features*/
              showCreateNewFeature && (
                <div className="row">
                  <form onSubmit={handleSubmit(createNewFeature)}>
                    <div className="col-4 secondary-panel">
                      <label className="form-label" htmlFor="newFeature">
                        Select feature to add
                      </label>
                      <select
                        className="form-control"
                        id="newFeature"
                        {...register('type')}
                        onChange={(event) => {
                          setFeatureToCreate(event.target.value);
                        }}
                      >
                        <option></option>
                        {Object.keys(project.features.options).map((element) => (
                          <option key={element} value={element}>
                            {element}
                          </option>
                        ))}{' '}
                      </select>
                      <button className="btn btn-primary btn-validation">Create</button>

                      {selectedFeatureToCreate === 'dfm' && (
                        <div>
                          <div>
                            <label htmlFor="dfm_tfidf">TF-IDF</label>
                            <select id="dfm_tfidf" {...register('parameters.dfm_tfidf')}>
                              <option>True</option>
                              <option>False</option>
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
                              <option>False</option>
                              <option>True</option>
                            </select>
                          </div>
                          <div>
                            <label htmlFor="dfm_log">Log</label>
                            <select id="dfm_log" {...register('parameters.dfm_log')}>
                              <option>False</option>
                              <option>True</option>
                            </select>
                          </div>
                        </div>
                      )}
                    </div>
                  </form>
                </div>
              )
            }

            {/* Display cards for each feature*/}
            <div className="row">
              <h2 className="subsection">Computed features</h2>
              {availableFeatures.map((element) => (
                <div className="card text-bg-light m-2 text-center" style={{ width: '10rem' }}>
                  <div className="card-body">
                    <h5 className="card-title">{element as string}</h5>
                    <div>
                      <button
                        className="btn btn p-0"
                        onClick={() => {
                          deleteSelectedFeature(element as string);
                        }}
                      >
                        <MdOutlineDeleteOutline size={30} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}{' '}
            </div>
          </div>
          <div></div>
        </div>
      )}
    </ProjectPageLayout>
  );
};
