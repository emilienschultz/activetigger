import { ReactNode } from 'react';

import { components } from './generated/openapi';

/**
 * API data types
 * this file rename API data types from the generated ones to make our code cleaner
 * Types are generated in a components['schemas'] map which is hideous
 * we extract what we use little by little when needed
 */

export type UserModel = components['schemas']['UserModel'];

export type ProjectModel = components['schemas']['ProjectModel'];
export type ProjectDataModel = components['schemas']['ProjectDataModel'];
export type ProjectStateModel = components['schemas']['StateModel'];
export type ElementOutModel = components['schemas']['ElementOutModel'];

export type AvailableProjectsModel = {
  created_by: string;
  created_at: string;
  parameters: ProjectModel;
};
export type LoginParams = components['schemas']['Body_login_for_access_token_token_post'];

/**
 * Notifications
 */
export interface NotificationData {
  title?: ReactNode;
  message: ReactNode;
  type: 'success' | 'info' | 'warning' | 'error';
}

export type NotificationType = NotificationData & { id: number; createdAt: Date };

export type SchemeModel = components['schemas']['SchemeModel'];

export type FeatureModel = components['schemas']['FeatureModel'];

export type RequestNextModel = components['schemas']['NextInModel'];

export type AnnotationModel = components['schemas']['AnnotationModel'];

export type SimpleModelModel = components['schemas']['SimpleModelModel'];

export type UsersServerModel = components['schemas']['UsersServerModel'];

export type BertModelParametersModel = components['schemas']['BertModelParametersModel'];

export interface FeatureDfmParameters {
  dfm_tfidf: string;
  ngrams: number;
  dfm_ngrams: number;
  dfm_min_term_freq: number;
  dfm_max_term_freq: number;
  dfm_norm: string;
  dfm_log: string;
}

export interface FeatureRegexParameters {
  value: string;
}

export interface FeatureModelExtended {
  name: string;
  type: string;
  parameters: null | FeatureDfmParameters | FeatureRegexParameters;
}

export interface SelectionConfig {
  mode: string;
  sample: string;
  label?: string;
  frame?: number[];
  displayPrediction?: boolean;
  displayContext?: boolean;
  filter?: string;
}

export interface newBertModel {
  name?: string;
  base: string;
  parameters: BertModelParametersModel;
}
